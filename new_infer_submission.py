"""
PatchCore 추론 → 제출 CSV + 히트맵 (B 방식: crop 안 함, 소자 박스 안 패치만 점수)

전체 512 이미지를 PatchCore에 넣되, YOLOE가 찾은 소자 박스 안 패치만 점수에 사용.
→ 소자 위치가 달라도 무관, 박스 밖 배경 이물질은 점수에서 제외됨.
→ YOLOE가 소자를 못 찾으면 '소자 부재' → label 1.

점수는 max 단독(소자 박스 안 패치의 최댓값). 거리 계산은 GPU(torch.cdist).

출력:
  - submission.csv : id(확장자 제거), label(0/1)
  - vis/           : 히트맵 오버레이 (공통 색 기준, 소자 박스 표시)

사용법:
  python infer_submission.py \
      --bank memory_bank_semicon_yolo.npz \
      --img_dir Datasets/test \
      --ref_image Datasets/train/Train_000.png \
      --ref_box_file ref_boxes.npy \
      --out submission.csv --vis_dir vis --threshold 3.0
"""
import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from device_region import init_detector, detect_box, box_to_patch_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor:
    def __init__(self, input_size):
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.eval().to(device)
        self.model = model
        self.input_size = input_size
        self.features = {}
        model.layer2.register_forward_hook(self._hook("layer2"))
        model.layer3.register_forward_hook(self._hook("layer3"))

    def _hook(self, name):
        def fn(module, inp, out):
            self.features[name] = out
        return fn

    def preprocess(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.input_size, self.input_size):
            rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        arr = rgb.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    @torch.no_grad()
    def extract(self, img_tensor):
        self.features = {}
        _ = self.model(img_tensor.to(device))
        f2 = self.features["layer2"]
        f3 = self.features["layer3"]
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                              align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches.squeeze(0), (H, W)        # GPU 텐서 (H*W, C), (H,W)


def main(args):
    # 메모리뱅크
    print("메모리뱅크 로드")
    start = time.time()
    data = np.load(args.bank)
    bank = torch.from_numpy(data["memory_bank"]).float().to(device)  # (M,C)
    bank_input = int(data["input_size"])
    input_size = args.input_size if args.input_size > 0 else bank_input
    extractor = FeatureExtractor(input_size)
    print(f"로드 완료 {time.time()-start:.2f}s  (bank {tuple(bank.shape)}, input {input_size})")

    # 소자 검출기 (학습과 같은 ref_box_file 사용 → 동일 예시 박스 보장)
    init_detector(args.ref_image, ref_box_file=args.ref_box_file,
                  model_path=args.yolo_model, conf=args.yolo_conf)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.img_dir, e))
    paths = sorted(paths)
    print(f"추론 대상 {len(paths)}장")

    rows = {}        # id -> label
    scores_all = []
    vis_items = []
    patch_vmin, patch_vmax = np.inf, -np.inf
    n_absent = 0

    for k, path in enumerate(paths):
        img_id = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            rows[img_id] = 1
            continue

        # 소자 검출: 실패하면 부재 → label 1
        box = detect_box(img)
        if box is None:
            rows[img_id] = 1
            n_absent += 1
            if args.vis_dir:    # 부재도 히트맵에 표시(박스 없음)
                vis_items.append((path, img_id, None, None, None))
            continue

        t = extractor.preprocess(img)
        patches, (H, W) = extractor.extract(t)         # (H*W, C) GPU
        with torch.no_grad():
            d = torch.cdist(patches, bank)             # (H*W, M)
            nn = d.topk(args.n_neighbors, dim=1, largest=False).values
            patch_scores = nn.mean(dim=1)
        patch_scores = patch_scores.cpu().numpy()      # (H*W,)

        # 소자 박스 안 패치만 선택
        m = box_to_patch_mask(box, img.shape[:2], (H, W), input_size)
        sel = patch_scores[m] if m.sum() > 0 else patch_scores

        score = float(sel.max())                       # max 단독
        scores_all.append(score)
        rows[img_id] = 1 if score > args.threshold else 0

        if args.vis_dir:
            amap = patch_scores.reshape(H, W)          # 전체맵(시각화는 전체)
            box_feat = m.reshape(H, W)
            patch_vmin = min(patch_vmin, sel.min())
            patch_vmax = max(patch_vmax, sel.max())
            vis_items.append((path, img_id, score, amap, box_feat))

        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(paths)}")

    # ── 히트맵 저장 (소자 박스 안만 색칠, 공통 색 기준) ──
    if args.vis_dir and vis_items:
        os.makedirs(args.vis_dir, exist_ok=True)
        rng = max(patch_vmax - patch_vmin, 1e-6)
        for path, img_id, score, amap, box_feat in vis_items:
            img = cv2.resize(cv2.imread(path), (input_size, input_size))
            if score is None:      # 소자 부재
                cv2.putText(img, "NO-DEVICE (label 1)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(args.vis_dir, f"{img_id}.png"), img)
                continue

            amap_r = cv2.resize(amap, (input_size, input_size))
            box_r = cv2.resize(box_feat.astype(np.uint8), (input_size, input_size),
                               interpolation=cv2.INTER_NEAREST) > 0
            norm = np.clip((amap_r - patch_vmin) / rng, 0, 1)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            overlay = img.copy()
            # 소자 박스 안만 히트맵, 밖은 원본 유지
            overlay[box_r] = cv2.addWeighted(img, 0.5, heat, 0.5, 0)[box_r]
            # 박스 경계 그리기
            ys, xs = np.where(box_r)
            if len(xs):
                cv2.rectangle(overlay, (xs.min(), ys.min()),
                              (xs.max(), ys.max()), (0, 255, 0), 2)

            label = rows[img_id]
            txt = f"{score:.2f} {'ANOMALY' if label else 'normal'}"
            color = (0, 0, 255) if label else (0, 255, 0)
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
            cv2.imwrite(os.path.join(args.vis_dir, f"{img_id}.png"), overlay)
        print(f"히트맵 저장 → {args.vis_dir}/")

    # ── submission.csv ──
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for img_id in sorted(rows):
            w.writerow([img_id, rows[img_id]])

    n_anom = sum(rows.values())
    print(f"\n저장 완료 → {args.out}  (총 {len(rows)}개)")
    print(f"이상(1): {n_anom} / 정상(0): {len(rows)-n_anom}  (소자부재 {n_absent} 포함)")
    if scores_all:
        s = np.array(scores_all)
        for p in (50, 70, 80, 90, 95):
            print(f"  상위 {100-p:2d}% 지점 th={np.percentile(s, p):.2f}")
        print(f"점수 min={s.min():.2f} median={np.median(s):.2f} "
              f"max={s.max():.2f}  (현재 threshold={args.threshold})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="new_memory_bank_semicon.npz")
    p.add_argument("--img_dir", default="Datasets/test", help="테스트 폴더(crop 안 함)")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--vis_dir", default="vis4", help="히트맵 폴더 (빈 문자열이면 끔)")
    p.add_argument("--threshold", type=float, default=3.2, help="이상 판정 임계값")
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--n_neighbors", type=int, default=1)
    # 소자 영역(YOLOE) — 학습과 같은 ref_box_file 써야 함
    p.add_argument("--ref_image", required=True, help="예시 박스 지정용 reference")
    p.add_argument("--ref_box_file", default="ref_boxes.npy")
    p.add_argument("--yolo_model", default="yoloe-11s-seg.pt")
    p.add_argument("--yolo_conf", type=float, default=0.1)
    args = p.parse_args()
    main(args)

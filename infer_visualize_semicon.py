"""
PatchCore 추론 → 히트맵 오버레이 이미지 저장 (512x512 전체 이미지 단위)
train_patchcore.py가 저장한 memory_bank.npz를 불러와,
각 테스트 이미지의 이상 위치를 히트맵으로 덧씌워 한 폴더에 점수순으로 저장한다.

- 결과 폴더: 파일명이 "<순위>_<점수>_<원본명>" 이라 탐색기에서 정렬하면
  가장 이상한 이미지가 맨 위에 온다. 위에서부터 보며 경계선을 정하면 된다.
- submission.csv 도 함께 생성 (threshold 기준 0/1)

의존성: torch, torchvision, numpy, scikit-learn, opencv-python
사용법:
    python infer_visualize.py --bank memory_bank.npz --img_dir test \
        --vis_dir vis --out submission.csv --threshold 3.0
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -------------------------------------------------------------------------------
# 특징 추출기 (학습 때와 동일 구조)
# -------------------------------------------------------------------------------
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
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t

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
        return patches.squeeze(0), (H, W)   # GPU 텐서 (H*W, C)


def main(args):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    # 메모리뱅크 로드
    print("메모리뱅크 로드 시작")
    start = time.time()
    data = np.load(args.bank)
    memory_bank = data["memory_bank"]
    bank_input_size = int(data["input_size"])

    input_size = args.input_size if args.input_size > 0 else bank_input_size
    if input_size != bank_input_size:
        print(f"[주의] 뱅크 학습 크기({bank_input_size}) != 추론 크기({input_size}).")

    # 메모리뱅크를 GPU 텐서로 상주 → cdist로 거리 계산
    bank_t = torch.from_numpy(memory_bank).float().to(device)  # (M, C)
    extractor = FeatureExtractor(input_size)
    print(f"로드 완료 {time.time() - start:.2f}s  "
          f"(bank {memory_bank.shape}, input {input_size})")

    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.img_dir, e))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"이미지를 찾지 못함: {args.img_dir}")
    print(f"추론 대상 {len(paths)}개")

    # ── 1차 패스: 점수 + 이상맵 수집 (히트맵 색을 공통 기준으로 칠하기 위함) ──
    results = []  # dict: path, fname, score, amap(H,W)
    patch_vmin, patch_vmax = np.inf, -np.inf
    for k, path in enumerate(paths):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[읽기실패] {fname} → 건너뜀")
            results.append({"path": path, "fname": fname, "score": 0.0,
                            "amap": None})
            continue

        t = extractor.preprocess(img)
        patches, (H, W) = extractor.extract(t)     # GPU 텐서 (H*W, C)

        # 각 패치 → 메모리뱅크 최근접 k개 거리 평균 (전부 GPU)
        with torch.no_grad():
            d = torch.cdist(patches, bank_t)       # (H*W, M)
            knn = d.topk(args.n_neighbors, dim=1, largest=False).values
            patch_scores = knn.mean(dim=1)         # (H*W,)
        patch_scores = patch_scores.cpu().numpy()

        score = float(patch_scores.max())          # 이미지 점수
        amap = patch_scores.reshape(H, W)

        patch_vmin = min(patch_vmin, patch_scores.min())
        patch_vmax = max(patch_vmax, patch_scores.max())
        results.append({"path": path, "fname": fname, "score": score,
                        "amap": amap})

        if (k + 1) % 20 == 0:
            print(f"  추론 {k + 1}/{len(paths)}")

    # ── 파일명 순 정렬 (Test_001, Test_002, ...) ──
    results.sort(key=lambda r: r["fname"])

    # ── 2차 패스: 히트맵 오버레이 저장 (공통 vmin/vmax로 정규화) ──
    os.makedirs(args.vis_dir, exist_ok=True)
    rng = max(patch_vmax - patch_vmin, 1e-6)
    for r in results:
        img = cv2.imread(r["path"])
        if img is None or r["amap"] is None:
            continue
        img = cv2.resize(img, (input_size, input_size))

        # 이상맵 → 공통 스케일 0~255 → 컬러맵 → 오버레이
        amap = cv2.resize(r["amap"], (input_size, input_size))
        norm = np.clip((amap - patch_vmin) / rng, 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

        # 점수/판정 텍스트
        label = 1 if r["score"] > args.threshold else 0
        txt = f"{r['score']:.2f} {'ANOMALY' if label else 'normal'}"
        color = (0, 0, 255) if label else (0, 255, 0)
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2)

        # 원본 파일명 그대로 저장 (Test_001.jpg ...)
        cv2.imwrite(os.path.join(args.vis_dir, r["fname"]), overlay)

    # ── submission.csv (원본 파일명 순서로 저장) ──
    rows = []
    for r in sorted(results, key=lambda x: x["fname"]):
        img_id = os.path.splitext(r["fname"])[0]   # 확장자 제거 (예: abc.jpg → abc)
        label = 1 if r["score"] > args.threshold else 0
        rows.append((img_id, label))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(rows)

    n_anom = sum(lbl for _, lbl in rows)
    s = np.array([r["score"] for r in results])
    print(f"\n히트맵 저장 → {args.vis_dir}/  ({len(results)}장)")
    print(f"submission 저장 → {args.out}")
    print(f"이상(1): {n_anom} / 정상(0): {len(rows) - n_anom}")
    print(f"점수 분포  min={s.min():.2f}  median={np.median(s):.2f}  "
          f"max={s.max():.2f}   (threshold={args.threshold})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="memory_bank_semicon.npz", help="메모리뱅크 경로")
    p.add_argument("--img_dir", default="./Datasets/test", help="테스트 이미지 폴더")
    p.add_argument("--vis_dir", default="vis", help="히트맵 저장 폴더")
    p.add_argument("--out", default="submission.csv", help="출력 CSV")
    p.add_argument("--threshold", type=float, default=3.0, help="이상 판정 임계값")
    p.add_argument("--input_size", type=int, default=512,
                   help="추론 입력 크기 (0이면 뱅크 학습 크기)")
    p.add_argument("--n_neighbors", type=int, default=1, help="최근접 이웃 수")
    args = p.parse_args()
    main(args)
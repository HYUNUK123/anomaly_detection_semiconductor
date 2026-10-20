"""
PatchCore 학습 스크립트
정상 이미지로 메모리뱅크를 구축하고 파일로 저장한다.
(추론은 저장된 memory_bank.npz를 불러와서 수행)

의존성: torch, torchvision, numpy, scikit-learn, pillow
사용법:
    python train_patchcore.py --train_dir dataset/train/good --save memory_bank.npz
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 입력 이미지 크기 (추론 때도 동일해야 함)
INPUT_SIZE = 512


# ──────────────────────────────────────────────
# 1. 특징 추출기: WideResNet50의 layer2, layer3 중간 feature
# ──────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, input_size=INPUT_SIZE):
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.eval().to(device)
        self.model = model

        self.features = {}
        model.layer2.register_forward_hook(self._hook("layer2"))
        model.layer3.register_forward_hook(self._hook("layer3"))

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # crop 없이 전체 보존
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, name):
        def fn(module, inp, out):
            self.features[name] = out
        return fn

    @torch.no_grad()
    def extract(self, img_tensor):
        """img_tensor: (B, 3, H, W) → patch features (B, H'*W', C), (H', W')"""
        self.features = {}
        _ = self.model(img_tensor.to(device))

        f2 = self.features["layer2"]  # (B, 512,  h/8,  w/8)
        f3 = self.features["layer3"]  # (B, 1024, h/16, w/16)

        # layer3를 layer2 해상도로 업샘플 후 채널 방향 concat
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                              align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)  # (B, 1536, h', w')

        # 3x3 이웃 평균 풀링 (locally aware patch feature)
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches.cpu(), (H, W)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0)


# ──────────────────────────────────────────────
# 2. Coreset subsampling: 메모리뱅크 압축
#    전체 패치 중 서로 멀리 떨어진 대표 샘플만 greedy하게 선택
# ──────────────────────────────────────────────
def coreset_subsample(features, sampling_ratio=0.1, proj_dim=128,
                      max_pool=50000, seed=42):
    """features: (N, C) numpy → 대표 (M, C) 반환 (GPU 가속 greedy FPS)

    max_pool: coreset 입력 패치 상한. 이보다 많으면 먼저 무작위로 솎아
              반복당 거리 계산을 가볍게 한다 (속도↑, 품질 거의 동일).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ① coreset 전 랜덤 축소: 전체 패치가 너무 많으면 대표만 남김
    if max_pool and len(features) > max_pool:
        sel = rng.choice(len(features), max_pool, replace=False)
        features = features[sel]
        print(f"  패치 {len(sel)}개로 랜덤 축소 (coreset 가속)")

    N = len(features)
    n_samples = max(1, int(N * sampling_ratio))

    feats = torch.from_numpy(features).float().to(device)  # (N, C)

    # 랜덤 투영으로 차원 축소 (GPU, Gaussian) — 거리 근사 보존
    if proj_dim and feats.shape[1] > proj_dim:
        R = torch.randn(feats.shape[1], proj_dim, device=device)
        R /= R.norm(dim=0, keepdim=True)
        proj = feats @ R                                   # (N, proj_dim)
    else:
        proj = feats

    # ||x||² 미리 계산 → 매 반복 내적만으로 거리 산출
    #   dist²(x, c) = ||x||² + ||c||² - 2·x·c
    sq_norms = (proj * proj).sum(dim=1)                    # (N,)

    start = int(torch.randint(N, (1,)).item())
    selected = torch.empty(n_samples, dtype=torch.long, device=device)
    selected[0] = start

    c = proj[start]                                        # (proj_dim,)
    min_d2 = sq_norms + sq_norms[start] - 2.0 * (proj @ c)  # (N,) 거리²

    for i in range(1, n_samples):
        idx = torch.argmax(min_d2)                         # GPU에 유지 (동기화 X)
        selected[i] = idx
        c = proj[idx]
        new_d2 = sq_norms + sq_norms[idx] - 2.0 * (proj @ c)
        min_d2 = torch.minimum(min_d2, new_d2)             # 최근접 거리 갱신
        if (i + 1) % 2000 == 0:
            print(f"  coreset {i + 1}/{n_samples}")

    sel_idx = selected.cpu().numpy()                       # 마지막에 한 번만 CPU로
    return features[sel_idx]


# ──────────────────────────────────────────────
# 3. 학습 (메모리뱅크 구축)
# ──────────────────────────────────────────────
def train(train_dir, save_path, sampling_ratio, input_size, max_pool=50000,
          ref_image=None, ref_box_file="ref_boxes.npy",
          yolo_model="yoloe-11s-seg.pt", yolo_conf=0.1):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    train_paths = []
    for e in exts:
        train_paths += glob.glob(os.path.join(train_dir, "**", e), recursive=True)
    train_paths = sorted(train_paths)

    if not train_paths:
        raise FileNotFoundError(f"학습 이미지를 찾지 못함: {train_dir}")
    print(f"정상 이미지 {len(train_paths)}장 로드")

    extractor = FeatureExtractor(input_size=input_size)

    # 소자 영역 검출기 준비 (전체 이미지 중 소자 박스 안 패치만 학습)
    use_region = bool(ref_image)
    if use_region:
        import cv2
        from device_region import init_detector, detect_box, box_to_patch_mask
        init_detector(ref_image, ref_box_file=ref_box_file,
                      model_path=yolo_model, conf=yolo_conf)
        print("소자 박스 안 패치만 메모리뱅크에 사용")

    all_patches = []
    skipped = 0
    for i, path in enumerate(train_paths):
        t = extractor.load_image(path)
        patches, (H, W) = extractor.extract(t)       # (1, H*W, C)
        p = patches.squeeze(0).numpy()               # (H*W, C)

        if use_region:
            img = cv2.imread(path)
            box = detect_box(img)
            if box is None:                          # 소자 미검출 → 학습서 제외
                skipped += 1
                continue
            m = box_to_patch_mask(box, img.shape[:2], (H, W), input_size)
            p = p[m]                                 # 소자 박스 안 패치만

        all_patches.append(p)
        if (i + 1) % 20 == 0:
            print(f"  feature 추출 {i + 1}/{len(train_paths)}")

    all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    print(f"전체 패치: {all_patches.shape}  (feature map {H}x{W}, 미검출 {skipped}장 제외)")

    print("coreset subsampling 시작...")
    memory_bank = coreset_subsample(all_patches, sampling_ratio,
                                    max_pool=max_pool)
    print(f"메모리뱅크: {memory_bank.shape} "
          f"(원본의 {len(memory_bank) / len(all_patches) * 100:.1f}%)")

    np.savez(save_path,
             memory_bank=memory_bank,
             input_size=input_size,
             feat_hw=np.array([H, W]))
    print(f"저장 완료 → {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="Datasets/train",
                   help="정상 이미지 폴더 (하위 폴더까지 재귀 탐색)")
    p.add_argument("--save", default="new_memory_bank_semicon.npz",
                   help="메모리뱅크 저장 경로")
    p.add_argument("--ratio", type=float, default=0.1,
                   help="coreset 샘플링 비율 (0.01~0.25 권장)")
    p.add_argument("--max_pool", type=int, default=200000,
                   help="coreset 입력 패치 상한. 초과분은 랜덤 축소 (속도↑)")
    p.add_argument("--input_size", type=int, default=INPUT_SIZE,
                   help="입력 이미지 리사이즈 크기")
    # ── 소자 영역(YOLOE) : 전체 이미지 중 소자 박스 안 패치만 학습 ──
    p.add_argument("--ref_image", default=None,
                   help="소자 예시 박스를 지정할 reference 이미지. "
                        "지정하면 소자 박스 안 패치만 사용(crop 안 함). 없으면 전체 패치")
    p.add_argument("--ref_box_file", default="ref_boxes.npy",
                   help="예시 박스 저장/재사용 파일 (학습·추론 공유)")
    p.add_argument("--yolo_model", default="yoloe-11s-seg.pt")
    p.add_argument("--yolo_conf", type=float, default=0.1)
    args = p.parse_args()

    train(args.train_dir, args.save, args.ratio, args.input_size, args.max_pool,
          args.ref_image, args.ref_box_file, args.yolo_model, args.yolo_conf)

"""
공용 모듈: YOLOE visual prompt로 소자 박스를 검출하고,
feature map (H,W) 위에서 "소자 박스 안 패치"만 True인 마스크를 만든다.

학습(train)과 추론(infer) 양쪽에서 이걸 똑같이 써서,
전체 이미지를 PatchCore에 넣되 소자 박스 안 패치만 점수/메모리뱅크에 사용한다.
(crop 하지 않음 → 소자 위치가 달라도 무관, 배경 이물질은 박스 밖이라 제외됨)

흐름:
  1) reference 이미지에서 소자 예시 박스를 1회 지정 → ref_boxes.npy 저장/재사용
  2) 각 이미지에서 YOLOE로 소자 박스 검출 (최고 신뢰도 1개)
  3) 그 박스를 feature map 해상도로 환산 → 박스 안 패치 마스크 반환
     (검출 실패 시 None → 소자 부재)
"""
import os

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

_model = None
_ref_set = False


# ── 드래그로 예시 박스 지정 (최초 1회) ──
_drawing = False
_ix = _iy = -1
_boxes = []
_disp = None


def _on_mouse(event, x, y, flags, param):
    global _drawing, _ix, _iy, _boxes, _disp
    base = param["img"]
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _ix, _iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _disp = base.copy()
        for b in _boxes:
            cv2.rectangle(_disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.rectangle(_disp, (_ix, _iy), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        x1, y1 = min(_ix, x), min(_iy, y)
        x2, y2 = max(_ix, x), max(_iy, y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            _boxes.append([x1, y1, x2, y2])
        _disp = base.copy()
        for b in _boxes:
            cv2.rectangle(_disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)


def _pick_boxes(ref_path):
    global _disp, _boxes
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"reference 못 읽음: {ref_path}")
    _disp = img.copy()
    win = "drag device box  (s=save, r=reset, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse, {"img": img})
    while True:
        cv2.imshow(win, _disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("s") and _boxes:
            break
        elif key == ord("r"):
            _boxes = []
            _disp = img.copy()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("취소됨")
    cv2.destroyAllWindows()
    return np.array(_boxes, dtype=float)


def init_detector(ref_path, ref_box_file="ref_boxes.npy",
                  model_path="yoloe-11s-seg.pt", conf=0.1):
    """YOLOE를 로드하고 예시 박스를 세팅한다. (학습/추론 시작 시 1회 호출)

    ref_box_file이 있으면 그 박스를 재사용, 없으면 드래그로 지정 후 저장.
    → 학습과 추론이 '같은 예시 박스'를 쓰도록 보장한다.
    """
    global _model, _ref_set
    _model = YOLOE(model_path)

    if os.path.exists(ref_box_file):
        ref_boxes = np.load(ref_box_file)
        print(f"예시 박스 재사용: {ref_box_file}  {ref_boxes.tolist()}")
    else:
        ref_boxes = _pick_boxes(ref_path)
        np.save(ref_box_file, ref_boxes)
        print(f"예시 박스 저장: {ref_box_file}  {ref_boxes.tolist()}")

    visual_prompts = {"bboxes": ref_boxes,
                      "cls": np.zeros(len(ref_boxes), dtype=int)}
    _ = _model.predict(ref_path, refer_image=ref_path,
                       visual_prompts=visual_prompts,
                       predictor=YOLOEVPSegPredictor,
                       conf=conf, verbose=False)
    _ref_set = True
    _model._conf = conf


def detect_box(img_bgr):
    """이미지에서 소자 박스(최고 신뢰도 1개)를 원본 좌표로 반환. 없으면 None"""
    assert _ref_set, "init_detector를 먼저 호출하세요"
    # YOLOE predict는 파일경로/배열 모두 가능. 배열로 바로 전달
    res = _model.predict(img_bgr, conf=_model._conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    return res.boxes.xyxy[best].cpu().numpy()   # [x1,y1,x2,y2] (원본 좌표)


def box_to_patch_mask(box, orig_hw, feat_hw, input_size, pad_ratio=0.05):
    """원본 박스 → feature map (H,W)에서 박스 안 패치 True 마스크 (H*W,) 반환.

    이미지가 input_size로 리사이즈된 뒤 feature map이 되므로,
    박스 좌표도 원본 → input_size → feature map 순으로 환산한다.
    """
    oh, ow = orig_hw
    H, W = feat_hw
    x1, y1, x2, y2 = box

    # 살짝 패딩 (박스가 소자를 조금 작게 잡아도 가장자리 결함 포함)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad_ratio; x2 += bw * pad_ratio
    y1 -= bh * pad_ratio; y2 += bh * pad_ratio

    # 원본 → feature map 스케일 (리사이즈는 비율 환산이라 중간 input_size 생략 가능)
    fx1 = int(np.floor(x1 / ow * W)); fx2 = int(np.ceil(x2 / ow * W))
    fy1 = int(np.floor(y1 / oh * H)); fy2 = int(np.ceil(y2 / oh * H))
    fx1 = max(0, fx1); fy1 = max(0, fy1)
    fx2 = min(W, fx2); fy2 = min(H, fy2)

    mask = np.zeros((H, W), dtype=bool)
    if fx2 > fx1 and fy2 > fy1:
        mask[fy1:fy2, fx1:fx2] = True
    return mask.ravel()      # (H*W,)

"""ROI 박스를 FoundationPose에 넘길 마스크로 정제하는 유틸리티."""

import cv2
import numpy as np


def mask_from_box(shape, depth, depth_scale, roi, refine=True, band=0.15):
    """ROI 박스 -> 마스크. 마우스 선택과 --init-roi 가 같은 정제 경로를 쓴다."""
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None

    mask = np.zeros(shape[:2], dtype=bool)
    mask[y:y + h, x:x + w] = True

    if refine:
        # 박스 안에서 뎁스 중앙값 근처만 남긴다. 박스 모서리에 딸려 들어온
        # 배경을 걷어내 register 품질이 눈에 띄게 좋아진다.
        d = depth[y:y + h, x:x + w].astype(np.float32) * depth_scale
        valid = d[(d > 0.05) & (d < 5.0)]
        if valid.size > 50:
            med = float(np.median(valid))
            dm = depth.astype(np.float32) * depth_scale
            mask &= (np.abs(dm - med) < band) & (dm > 0.05)
            # 자잘한 구멍/점 정리
            m8 = mask.astype(np.uint8)
            k = np.ones((5, 5), np.uint8)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
            if m8.sum() > 100:
                mask = m8.astype(bool)

    return mask if mask.sum() > 100 else None


def make_mask_from_roi(color, depth, depth_scale, refine=True, band=0.15):
    """마우스로 박스를 끌어 마스크를 만든다. Enter/Space 확정, c 취소."""
    win = "select object (drag -> ENTER)"
    roi = cv2.selectROI(win, color, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(win)
    return mask_from_box(color.shape, depth, depth_scale, roi,
                         refine=refine, band=band)

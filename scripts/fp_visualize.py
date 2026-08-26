"""pose를 화면에 그리는 함수들 (OBB 와이어프레임 / 좌표축)."""

import cv2
import numpy as np


def _project(K, pts_cam):
    """카메라 좌표계 3D 점(N,3) -> 픽셀 좌표(N,2). z<=0은 NaN."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    uv[z <= 1e-6] = np.nan
    return uv


def draw_posed_3d_box(img, K, ob_in_cam, bbox, color=(0, 255, 0), thickness=2):
    """bbox = [[xmin,ymin,zmin],[xmax,ymax,zmax]] (물체 중심 좌표계, m)."""
    mn, mx = np.asarray(bbox[0]), np.asarray(bbox[1])
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    R, t = ob_in_cam[:3, :3], ob_in_cam[:3, 3]
    uv = _project(K, (R @ corners.T).T + t)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        if np.isnan(uv[a]).any() or np.isnan(uv[b]).any():
            continue
        cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])),
                 color, thickness, cv2.LINE_AA)
    return img


def draw_xyz_axis(img, K, ob_in_cam, scale=0.06, thickness=3):
    """물체 좌표계 축을 그린다. x=빨강, y=초록, z=파랑 (BGR 이미지 기준)."""
    pts = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    R, t = ob_in_cam[:3, :3], ob_in_cam[:3, 3]
    uv = _project(K, (R @ pts.T).T + t)
    if np.isnan(uv).any():
        return img
    o = tuple(np.int32(uv[0]))
    for i, bgr in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)], start=1):
        cv2.line(img, o, tuple(np.int32(uv[i])), bgr, thickness, cv2.LINE_AA)
    return img

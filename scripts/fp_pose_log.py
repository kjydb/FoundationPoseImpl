"""4x4 pose를 JSON Lines로 기록/변환하는 유틸리티."""

import json
import time
from datetime import datetime

import numpy as np


def mat_to_quat(R):
    """3x3 회전행렬 -> 쿼터니언 (w, x, y, z). 짐벌락 없는 표현."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    """쿼터니언 (w, x, y, z) -> 3x3 회전행렬. 정규화 안 된 입력도 받는다
    (One Euro filter로 성분별로 스무딩한 뒤라 노름이 1에서 살짝 벗어날 수 있음)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    w, x, y, z = q
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def mat_to_euler_deg(R):
    """3x3 회전행렬 -> (roll, pitch, yaw) 도(deg). 화면 표시용, 짐벌락 근처에서만 근사."""
    m = np.asarray(R, dtype=np.float64)
    sy = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(m[2, 1], m[2, 2])
        pitch = np.arctan2(-m[2, 0], sy)
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:
        roll = np.arctan2(-m[1, 2], m[1, 1])
        pitch = np.arctan2(-m[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


class PoseLog:
    """pose를 JSON Lines로 흘려 쓴다. 디지털 트윈 쪽에서 그대로 읽어 쓰면 된다."""

    def __init__(self, path):
        now = datetime.now()
        full_path = path + now.strftime('%Y-%m-%d_%H:%M:%S') + '.jsonl'
        self.f = open(full_path, "w", encoding="utf-8")
        print(f"[log] pose 기록: {full_path}")

    def write(self, frame_id, pose, infer_ms):
        rec = {
            "frame_id": int(frame_id),
            "stamp": time.time(),
            "infer_ms": round(float(infer_ms), 2),
            "t": [round(float(v), 6) for v in pose[:3, 3]],
            "q_wxyz": [round(float(v), 6) for v in mat_to_quat(pose[:3, :3])],
            "T": [[round(float(v), 6) for v in row] for row in pose],
        }
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

"""One Euro filter (Casiez et al. 2012)로 pose 출력의 노이즈를 줄인다.

정지 상태에서 위치는 이미 노이즈가 작은데 회전만 눈에 띄게 흔들리는 걸 확인했고,
메쉬/텍스처/조명은 지금 당장 못 건드리는 제약이 있어서 pose 출력단 필터로
잔여 노이즈를 억제하기로 했다 (자세한 배경은 대화 맥락 참고).

핵심 성질: 속도(변화율)가 작을 때는 컷오프를 낮춰 강하게 스무딩하고, 속도가
커지면 컷오프를 같이 올려 거의 스무딩 없이 원신호를 따라간다. 정지/움직임을
따로 판별하는 로직 없이 이 동작이 자연스럽게 나온다.

**중요**: 여기서 만드는 필터링된 pose는 표시/로깅 등 후처리 전용이다.
FoundationPose 내부 추적 상태(est.pose_last)는 이 필터와 무관하게 매 프레임
원본(raw) 관측값으로만 갱신된다 -- register()/track()의 리턴값을 스무딩해서
다시 est에 먹이는 게 아니라, main 루프에서 그 리턴값을 받은 *다음*에 화면
표시/로깅용으로만 한 번 더 통과시키는 구조라서 트래커 자체의 피드백 루프에는
전혀 섞이지 않는다.
"""

import math

import numpy as np

from fp_pose_log import mat_to_quat, quat_to_mat


def _smoothing_factor(t_e, cutoff):
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(a, x, x_prev):
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """단일 스칼라 채널용 1€ 필터."""

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        """새 앵커(register 직후)에서 호출. 다음 호출은 원값을 그대로
        통과시키며 상태를 그 값으로 다시 채운다 -- 재초기화 직후의 점프를
        '빠른 움직임'으로 오인해 스무딩하는 걸 막는다."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = _smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PoseOneEuroFilter:
    """4x4 pose 스트림에 One Euro를 건다.

    위치(x,y,z)는 독립 채널 3개, 회전은 쿼터니언 성분(w,x,y,z) 독립 채널 4개에
    필터를 걸고 다시 단위 노름으로 정규화한다. 완전히 엄밀한 SLERP은 아니지만
    (성분별 근사), 지금 수준의 노이즈(정지 시 각도 std 1~2도대)에서는 실무적으로
    충분하고 구현이 훨씬 단순하다.

    쿼터니언은 q와 -q가 같은 회전이라, 필터에 넣기 전에 직전 프레임과 부호가
    반대면 뒤집어(hemisphere continuity) 필터의 속도 추정이 왜곡되지 않게 한다.
    """

    def __init__(self, pos_mincutoff=1.0, pos_beta=0.0,
                 rot_mincutoff=1.0, rot_beta=0.0, dcutoff=1.0,
                 filter_pos=True, filter_rot=True):
        self.filter_pos = filter_pos
        self.filter_rot = filter_rot
        # 꺼진 채널은 아예 안 만든다 -- 안 쓰는 필터를 매 프레임 불러서
        # 상태만 쌓는 걸 피한다.
        self._pos = ([OneEuroFilter(mincutoff=pos_mincutoff, beta=pos_beta,
                                    dcutoff=dcutoff) for _ in range(3)]
                    if filter_pos else None)
        self._quat = ([OneEuroFilter(mincutoff=rot_mincutoff, beta=rot_beta,
                                     dcutoff=dcutoff) for _ in range(4)]
                     if filter_rot else None)
        self._q_prev = None  # 연속성(부호) 판단용 직전 원본 쿼터니언

    def reset(self):
        """register 성공 직후(새 앵커) 호출. 그 프레임의 pose는 원값 그대로
        통과되고, 그다음 track 프레임부터 필터링이 걸린다."""
        if self._pos is not None:
            for f in self._pos:
                f.reset()
        if self._quat is not None:
            for f in self._quat:
                f.reset()
        self._q_prev = None

    def filter(self, pose, t):
        """pose: 4x4 np.ndarray (ob_in_cam). 같은 모양의 필터링된 pose를 돌려준다.

        filter_pos/filter_rot이 꺼진 채널은 원값을 그대로 통과시킨다."""
        pos = np.asarray(pose[:3, 3], dtype=np.float64)
        q = mat_to_quat(pose[:3, :3])

        if self._quat is not None:
            if self._q_prev is not None and np.dot(self._q_prev, q) < 0:
                q = -q
            self._q_prev = q

            q_f = np.array([f(v, t) for f, v in zip(self._quat, q)])
            n = np.linalg.norm(q_f)
            q_out = q_f / n if n > 1e-9 else q
        else:
            q_out = q

        if self._pos is not None:
            pos_out = np.array([f(v, t) for f, v in zip(self._pos, pos)])
        else:
            pos_out = pos

        out = np.eye(4)
        out[:3, :3] = quat_to_mat(q_out)
        out[:3, 3] = pos_out
        return out

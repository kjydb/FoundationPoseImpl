"""RealSense D455 캡처 래퍼 + 센서 옵션(노출/화이트밸런스/레이저파워) 설정."""

import cv2
import numpy as np
import pyrealsense2 as rs


def _find_sensor_by_name(dev, name):
    for s in dev.query_sensors():
        if s.get_info(rs.camera_info.name) == name:
            return s
    return None


def _set_manual_color(dev, exposure, white_balance):
    """AE/AWB를 끄고 고정값을 준다. 프레임마다 밝기·색이 바뀌면 RGB 입력이
    흔들려서 pose도 따라 흔들린다 -- 조명이 일정한 환경이 전제다."""
    if exposure is None and white_balance is None:
        return
    color = _find_sensor_by_name(dev, "RGB Camera")
    if color is None:
        print("[cam] !! RGB Camera 센서를 못 찾음 - exposure/white-balance 설정 건너뜀")
        return
    if exposure is not None:
        color.set_option(rs.option.enable_auto_exposure, 0)
        color.set_option(rs.option.exposure, float(exposure))
        print(f"[cam] 수동 노출: exposure={exposure} (100us 단위)")
    if white_balance is not None:
        color.set_option(rs.option.enable_auto_white_balance, 0)
        color.set_option(rs.option.white_balance, float(white_balance))
        print(f"[cam] 수동 화이트밸런스: {white_balance}K")


def _set_laser_power(dev, power):
    """IR 구조광 파워를 올려 유효 depth 픽셀 수를 늘린다."""
    if power is None:
        return
    depth_sensor = dev.first_depth_sensor()
    if not depth_sensor.supports(rs.option.laser_power):
        print("[cam] !! 이 장치는 laser_power를 지원하지 않음")
        return
    rng = depth_sensor.get_option_range(rs.option.laser_power)
    val = float(np.clip(power, rng.min, rng.max))
    depth_sensor.set_option(rs.option.laser_power, val)
    print(f"[cam] IR 레이저 파워: {val:.0f} (범위 {rng.min:.0f}~{rng.max:.0f})")


class D455:
    """align된 컬러/뎁스 프레임과 컬러 intrinsics를 내주는 얇은 래퍼.

    원격 버전과 다른 점: 추론이 캡처보다 느리므로 매 루프마다 큐에 쌓인
    프레임을 전부 비우고 **가장 최근 것만** 쓴다. 이걸 안 하면 화면이
    실제보다 몇 프레임 뒤처져 보인다.
    """

    def __init__(self, width=848, height=480, fps=30, filter_depth=True,
                 bag=None, repeat_bag=False, depth_min=0.2, depth_max=4.0,
                 exposure=None, white_balance=None, laser_power=None,
                 temporal_alpha=None):
        if rs is None:
            raise RuntimeError("pyrealsense2가 없다. pip install pyrealsense2")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if bag:
            # 녹화된 .bag 재생 (카메라 없이 재현 실험할 때)
            rs.config.enable_device_from_file(cfg, bag, repeat_playback=repeat_bag)
        else:
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.is_bag = bool(bag)

        if bag:
            # 실시간 재생을 끄면 프레임이 버려지지 않는다(추론이 느려도 안전).
            self.profile.get_device().as_playback().set_real_time(False)

        # 뎁스를 컬러 프레임에 맞춘다. 이걸 안 하면 FoundationPose에 넘기는
        # K(컬러 intrinsics)와 뎁스 픽셀이 서로 어긋나 pose가 통째로 틀어진다.
        self.align = rs.align(rs.stream.color)

        dev = self.profile.get_device()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()  # 보통 0.001

        if not bag:
            # bag 재생에는 실제 센서가 없어 옵션 자체가 없다.
            _set_manual_color(dev, exposure, white_balance)
            _set_laser_power(dev, laser_power)

        intr = (self.profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.width, self.height = intr.width, intr.height

        self.filter_depth = filter_depth
        if filter_depth:
            # threshold를 제일 먼저 건다. spatial/temporal보다 앞에 둬야
            # D455 min-Z(~0.5m) 언저리의 튀는 값이나 far-range 노이즈가
            # "유효한 이웃값"으로 스무딩에 섞여 들어가는 걸 막는다.
            self._threshold = rs.threshold_filter()
            self._threshold.set_option(rs.option.min_distance, depth_min)
            self._threshold.set_option(rs.option.max_distance, depth_max)

            self._to_disp = rs.disparity_transform(True)
            self._to_depth = rs.disparity_transform(False)

            # smooth_delta(엣지로 볼 depth 차이 임계값, 기본 20)를 낮춰서 물체-배경
            # 경계가 뭉개지지 않게 한다. 작은 물체/배경과 거리차가 크지 않을수록 더 낮출 것.
            self._spatial = rs.spatial_filter()
            self._spatial.set_option(rs.option.filter_smooth_delta, 15)

            # temporal은 과거 프레임을 끌어와 노이즈를 줄이는 대신, 물체가 빠르게
            # 움직이면 경계에 잔상(smear)을 남긴다. 물체가 거의 정적이면 기본값
            # (smooth_alpha=0.4)도 괜찮지만, 손으로 빠르게 움직이며 추적한다면
            # 최근 프레임 비중을 높여야 한다. 예:
            #   self._temporal.set_option(rs.option.filter_smooth_alpha, 0.7)
            self._temporal = rs.temporal_filter()
            if temporal_alpha is not None:
                self._temporal.set_option(rs.option.filter_smooth_alpha,
                                          float(temporal_alpha))
                print(f"[cam] temporal filter alpha = {temporal_alpha}")

            # mode 2 = "Nearest from around": 구멍을 배경(먼 값)이 아니라 전경
            # 쪽 값으로 채운다. 기본 mode 1("Farest from around")은 광택/검은
            # 표면 때문에 물체 위에 뚫린 구멍에 배경 depth가 새어 들어오기 쉽다.
            self._hole = rs.hole_filling_filter(2)

        if not bag:
            # 자동 노출이 안정될 때까지 몇 프레임 버린다.
            for _ in range(30):
                self.pipeline.wait_for_frames()

        if False:
            color = _find_sensor_by_name(dev, "RGB Camera")
            print("현재 exposure:", color.get_option(rs.option.exposure))
            print("현재 white_balance:", color.get_option(rs.option.white_balance))

    def _latest_frames(self):
        """큐를 비우고 가장 최근 프레임셋을 돌려준다."""
        frames = self.pipeline.wait_for_frames()
        if not self.is_bag:
            # poll_for_frames()는 (bool, frame) 튜플이 아니라 composite_frame
            # 하나를 돌려준다. try_wait_for_frames(0)이 진짜 (성공여부, 프레임)
            # 페어를 주는 non-blocking API다.
            while True:
                ok, newer = self.pipeline.try_wait_for_frames(timeout_ms=0)
                if not ok:
                    break
                frames = newer
        return frames

    def read(self):
        frames = self.align.process(self._latest_frames())
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if not cf or not df:
            return None, None
        if self.filter_depth:
            df = self._threshold.process(df)
            df = self._to_disp.process(df)
            df = self._spatial.process(df)
            df = self._temporal.process(df)
            df = self._to_depth.process(df)
            df = self._hole.process(df)
        color = np.asanyarray(cf.get_data())            # BGR uint8
        depth = np.asanyarray(df.get_data()).astype(np.uint16)
        return color, depth

    def close(self):
        self.pipeline.stop()


def downscale(color, depth, K, scale):
    """해상도를 낮출 때 K도 같이 줄여야 한다.

    로컬 실행에서는 대역폭이 아니라 **추론 속도**를 위해 쓴다. FoundationPose는
    crop 후 고정 크기로 리사이즈하므로 효과가 원격만큼 크진 않지만, 뎁스 필터와
    메모리 전송량이 줄어든다. VRAM 절감 효과도 작다 -- 피크는 해상도가 아니라
    자세 가설 개수가 만든다.
    """
    if scale == 1.0:
        return color, depth, K
    h, w = color.shape[:2]
    nw, nh = int(round(w * scale)), int(round(h * scale))
    color_s = cv2.resize(color, (nw, nh), interpolation=cv2.INTER_AREA)
    # 뎁스는 반드시 NEAREST. 보간하면 물체 경계에 없는 거리값이 생긴다.
    depth_s = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    Ks = K.copy()
    Ks[0, :] *= nw / w
    Ks[1, :] *= nh / h
    return color_s, depth_s, Ks

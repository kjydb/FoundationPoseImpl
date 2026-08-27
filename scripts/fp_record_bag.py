"""fp_record_bag.py — D455의 컬러+뎁스를 .bag으로 녹화한다.

fp_local_d455.py --bag으로 재생할 원본 데이터를 만드는 용도. FoundationPose를
쓰지 않는 순수 캡처 스크립트라 fp_pose_engine은 import하지 않고, 센서 옵션
설정 코드만 fp_camera.py에서 그대로 가져와 쓴다.

bag 재생에는 실제 센서가 없어 노출/화이트밸런스/레이저 파워를 나중에 바꿀 수
없다 (fp_camera.py의 D455가 bag일 때 이 설정을 건너뛰는 이유이기도 하다).
따라서 --exposure/--white-balance/--laser-power는 녹화 시점에 확정해서
찍어야 하며, fp_local_d455.py와 같은 이름의 인자를 그대로 쓴다 -- 나중에
재생할 때 같은 값을 주면(사실 줘도 무시되지만) 무엇으로 찍었는지 헷갈리지
않는다.

자동노출/자동WB가 켜져 있으면 켠 직후 몇 프레임은 밝기가 흔들린다. 이 안정화
구간이 녹화 파일 앞부분에 그대로 박히지 않도록, enable_record_to_file을 걸기
*전에* 별도 파이프라인으로 워밍업을 먼저 끝낸다.

실행 예
    python fp_record_bag.py --output ./bags/sample.db3 \
        --exposure 150 --white-balance 4600 --laser-power 150

키 (미리보기 창)
    q / ESC : 녹화 종료
"""

import argparse
import datetime
import os

import cv2
import numpy as np
import pyrealsense2 as rs

from fp_camera import _set_manual_color, _set_laser_power

# 최신 librealsense(2.55+)는 녹화 저장소가 rosbag2(SQLite)로 바뀌어
# enable_record_to_file()에 .db3 확장자를 요구한다 (안 맞으면
# "Output file must have .db3 extension" RuntimeError). 재생 쪽
# (enable_device_from_file, fp_camera.py의 --bag)은 확장자를 안 가리므로
# 기존에 만든 .bag 파일도 그대로 재생된다 -- 새로 녹화하는 파일만 .db3여야
# 한다.
RECORD_EXT = ".db3"


def _open_pipeline(width, height, fps, record_to=None):
    cfg = rs.config()
    if record_to:
        cfg.enable_record_to_file(record_to)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    return pipeline, profile


def main():
    ap = argparse.ArgumentParser(description="D455 -> .bag 녹화")
    ap.add_argument("--output", default=None,
                    help="저장할 경로 (기본: ./bags/%%Y-%%m-%%d_%%H:%%M:%%S.db3). "
                         f"librealsense가 확장자로 포맷을 가리므로 항상 "
                         f"{RECORD_EXT}로 저장된다 -- 다른 확장자를 주면 자동으로 "
                         f"바꾼다")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=None,
                    help="컬러 센서 수동 노출(100us 단위). 주면 자동노출을 끈다")
    ap.add_argument("--white-balance", type=int, default=None,
                    help="컬러 센서 수동 화이트밸런스(Kelvin). 주면 자동WB를 끈다")
    ap.add_argument("--laser-power", type=float, default=None,
                    help="IR 구조광 파워. 높일수록 유효 depth 픽셀이 늘지만 "
                         "너무 가까운 거리에서는 과포화될 수 있다")
    ap.add_argument("--warmup-frames", type=int, default=30,
                    help="자동노출 등이 안정될 때까지 녹화 시작 전에 버릴 프레임 수")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="이만큼 녹화하고 자동 종료 (0이면 무제한 -- q/ESC나 "
                         "Ctrl+C로 종료)")
    ap.add_argument("--no-window", action="store_true",
                    help="미리보기 창 없이 헤드리스로 녹화. --max-frames나 "
                         "Ctrl+C로만 종료할 수 있다")
    args = ap.parse_args()

    output = args.output
    if output is None:
        os.makedirs("./bags", exist_ok=True)
        output = os.path.join(
            "./bags",
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + RECORD_EXT)
    else:
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not output.endswith(RECORD_EXT):
            fixed = os.path.splitext(output)[0] + RECORD_EXT
            print(f"[record] !! librealsense는 녹화 파일이 {RECORD_EXT} 확장자여야 "
                  f"함 -- {output} 대신 {fixed}로 저장한다")
            output = fixed

    # ---- 워밍업 (녹화 전, 별도 파이프라인)
    warm_pipe, warm_profile = _open_pipeline(args.width, args.height, args.fps)
    dev = warm_profile.get_device()
    _set_manual_color(dev, args.exposure, args.white_balance)
    _set_laser_power(dev, args.laser_power)
    for _ in range(args.warmup_frames):
        warm_pipe.wait_for_frames()
    warm_pipe.stop()

    # ---- 녹화 시작 (설정이 확실히 유지되도록 다시 한번 적용)
    pipeline, profile = _open_pipeline(
        args.width, args.height, args.fps, record_to=output)
    dev = profile.get_device()
    _set_manual_color(dev, args.exposure, args.white_balance)
    _set_laser_power(dev, args.laser_power)

    print(f"[record] 녹화 시작: {output}")
    print(f"[record] 종료: {'--max-frames 도달' if args.max_frames else 'q/ESC 또는 Ctrl+C'}")

    win = "recording (q/ESC to stop)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            n += 1
            if not args.no_window:
                cf = frames.get_color_frame()
                if cf:
                    color = np.asanyarray(cf.get_data())
                    cv2.putText(color, f"REC {n} frames", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win, color)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[record] 중단")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[record] 종료. {n} 프레임 저장: {output}")


if __name__ == "__main__":
    main()

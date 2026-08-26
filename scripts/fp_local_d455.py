"""
fp_local_d455.py — 카메라와 FoundationPose가 **같은 컴퓨터**에 있을 때 쓰는 단일 프로세스 버전.

receiver_server.py(+ sender_d455.py)의 동작을 그대로 유지하되 ZeroMQ / JPEG·PNG
인코딩 계층을 전부 걷어냈다. 프레임은 numpy 배열 그대로 추정기에 들어간다.

하는 일
  1. RealSense D455에서 컬러/뎁스를 받아 컬러 프레임에 align
  2. 첫 프레임에서 마우스로 ROI를 끌어 마스크를 만들고 est.register()
  3. 이후 매 프레임 est.track_one()
  4. 4x4 pose로 mesh 오버레이(또는 OBB 박스) + 좌표축을 그려 화면에 표시 (원하면 파일로 기록)

FoundationPose 도커/콘다 환경 안에서 실행해야 한다. 그 환경에 pyrealsense2가
없다면 먼저 설치한다.
    pip install pyrealsense2 opencv-python

실행 예
    python fp_local_d455.py \
        --mesh /data/mybody/mybody.obj \
        --mesh-unit mm \
        --fp-root /opt/FoundationPose

VRAM이 빠듯할 때 (자세한 건 아래 'VRAM' 절 참고)
    python fp_local_d455.py --mesh ... --vram-probe \
        --view-subdiv 0 --inplane-step 120 --register-chunk 32 --no-cudnn-benchmark

키
    r : ROI 다시 잡고 재초기화 (추적이 발산했을 때)
    s : 현재 프레임/pose 스냅샷 저장
    q / ESC : 종료

메쉬 단위 주의: CAD가 mm면 --mesh-unit mm 를 준다. 이걸 틀리면 pose가
아예 수렴하지 않는다 (가장 흔한 실수).

VRAM
----
피크는 거의 전부 register() 단계에서 나온다. FoundationPose는 rot_grid의
모든 자세 가설(기본 252개)을 한 배치로 refiner/scorer에 밀어 넣고, scorer는
가설끼리 cross-attention까지 하므로 메모리가 가설 개수에 가파르게 비례한다.
반면 track_one()은 가설이 1개뿐이라 훨씬 적게 쓴다. 해상도의 영향은 작다 --
신경망은 crop 후 input_resize(보통 160x160)에서 돌기 때문이다.

줄이는 순서:
    1) --view-subdiv 0 --inplane-step 120   (252 -> 36 가설, 가장 큰 효과)
    2) --register-chunk 32                  (피크가 청크 크기에 거의 비례)
    3) --no-cudnn-benchmark                 (알고리즘 탐색 workspace 제거)
    4) PYTORCH_CUDA_ALLOC_CONF              (아래에서 자동 설정, 파편화 완화)
어디가 피크인지는 --vram-probe 로 먼저 측정할 것.

파일 구성
---------
역할별로 나뉜 이웃 모듈들을 그대로 import해서 쓴다 (같은 scripts/ 폴더에 있어야 함):
    fp_vram.py         GPU 메모리 사용량 출력
    fp_pose_engine.py  FoundationPose 래퍼(PoseEngine) + mesh 오버레이 렌더링
    fp_camera.py       RealSense D455 캡처 + 센서 옵션(노출/화이트밸런스/레이저)
    fp_mask.py         ROI -> 마스크 정제
    fp_visualize.py    OBB 박스 / 좌표축 그리기
    fp_pose_log.py     pose를 JSON Lines로 기록
이 파일 자체는 argparse + 메인 루프만 담당한다.
"""

import argparse
import os
import time

# torch가 import되기 *전에* 설정해야 효과가 있다. 캐싱 할당자가 큰 블록을
# 잘게 쪼개 쥐고 있는 파편화를 줄여, nvidia-smi에 찍히는 reserved 값을 낮춘다.
# fp_pose_engine이 torch/FoundationPose를 지연 import하므로 여기서 제일 먼저 설정한다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

from fp_pose_engine import PoseEngine
from fp_camera import D455, downscale
from fp_mask import mask_from_box, make_mask_from_roi
from fp_visualize import draw_posed_3d_box, draw_xyz_axis
from fp_pose_log import mat_to_euler_deg, PoseLog
from fp_one_euro import PoseOneEuroFilter


def main():
    ap = argparse.ArgumentParser(
        description="D455 + FoundationPose 로컬 단일 프로세스 추적")

    ap.add_argument("--log_level", type=str, default='info')

    # ---- 물체
    ap.add_argument("--mesh", required=True, help="추적할 물체의 mesh (.obj 등)")
    ap.add_argument("--mesh-unit", default="m", choices=["m", "cm", "mm"],
                    help="mesh 파일의 길이 단위. 틀리면 pose가 수렴하지 않는다")
    ap.add_argument("--fp-root", default=None,
                    help="FoundationPose 저장소 루트 (estimater.py가 있는 곳)")
    ap.add_argument("--symmetry-axis", default=None, choices=["x", "y", "z"],
                    help="물체가 이산 회전대칭이면 그 축(mesh 로컬 좌표계 기준). "
                         "주면 --symmetry-order와 함께 FoundationPose에 대칭 "
                         "변환을 등록해 그 축 둘레 회전 노이즈를 줄인다")
    ap.add_argument("--symmetry-order", type=int, default=2,
                    help="대칭 차수. 2=180도 간격(2겹), 4=90도 간격(4겹) 등. "
                         "--symmetry-axis를 줬을 때만 쓰인다")

    # ---- 카메라
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="처리 전 해상도 배율 (0.5면 절반). K도 함께 조정됨")
    ap.add_argument("--bag", default=None,
                    help="카메라 대신 녹화된 .bag 파일을 재생한다")
    ap.add_argument("--repeat-bag", action="store_true",
                    help=".bag을 반복 재생")
    ap.add_argument("--no-depth-filter", action="store_true")
    ap.add_argument("--no-mask-refine", action="store_true")
    ap.add_argument("--exposure", type=int, default=None,
                    help="컬러 센서 수동 노출(100us 단위). 주면 자동노출을 끈다. "
                         "조명이 일정한 환경에서 프레임간 밝기 흔들림을 없애 "
                         "정지 노이즈를 줄이는 데 도움된다")
    ap.add_argument("--white-balance", type=int, default=None,
                    help="컬러 센서 수동 화이트밸런스(Kelvin). 주면 자동WB를 끈다")
    ap.add_argument("--laser-power", type=float, default=None,
                    help="IR 구조광 파워. 높일수록 유효 depth 픽셀이 늘지만 "
                         "너무 가까운 거리에서는 과포화될 수 있다")
    ap.add_argument("--temporal-alpha", type=float, default=None,
                    help="depth temporal filter의 smooth_alpha. 기본은 SDK 기본값"
                         "(0.4)을 그대로 둔다. 낮출수록 정지 시 노이즈는 줄지만 "
                         "물체가 움직이면 경계에 잔상(smear)이 생기니, 물체가 "
                         "계속 정지해 있는 경우에만 낮출 것")

    # ---- 추정
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--view-subdiv", type=int, default=None, choices=[0, 1, 2],
                    help="초기 자세 가설의 시점 개수. 0=12뷰(빠름/가벼움), "
                         "1=42뷰(FoundationPose 기본), 2=162뷰(느림/정밀)")
    ap.add_argument("--inplane-step", type=int, default=60,
                    help="인플레인 회전 간격(도). 60=6단계(기본), 90=4단계, "
                         "120=3단계. 키울수록 가설이 줄어 빨라지고 VRAM도 준다")
    ap.add_argument("--no-warmup", action="store_true",
                    help="시작 시 더미 register로 예열하지 않는다")
    ap.add_argument("--zfar", type=float, default=2.0,
                    help="이보다 먼 뎁스는 0으로 버린다 (m)")

    # ---- VRAM
    ap.add_argument("--register-chunk", type=int, default=0,
                    help="register를 자세 가설 N개씩 나눠 돌린다(0=끄기). "
                         "VRAM 피크가 거의 N에 비례해 내려간다. 대신 느려지고, "
                         "조각 안에서만 순위를 매기므로 결과가 미세하게 달라질 수 있다")
    ap.add_argument("--no-cudnn-benchmark", action="store_true",
                    help="cuDNN 알고리즘 자동 탐색을 끈다. 조금 느려지지만 "
                         "탐색 중 workspace 할당이 사라져 피크가 내려간다")
    ap.add_argument("--vram-probe", action="store_true",
                    help="단계별 VRAM(alloc/reserved/peak)을 찍는다. "
                         "어느 단계가 피크를 만드는지 여기서 판별한다")

    # ---- 출력
    ap.add_argument("--vis-mode", default="mesh", choices=["box", "mesh", "both"],
                    help="화면 표시 방식. box=OBB 와이어프레임(기존), "
                         "mesh=실제 mesh를 pose에 맞춰 렌더링해 반투명하게 씌움, "
                         "both=둘 다")
    ap.add_argument("--overlay-alpha", type=float, default=0.6,
                    help="--vis-mode mesh/both일 때 렌더링된 mesh의 불투명도 (0~1)")
    ap.add_argument("--filter-type", default="none", choices=["none", "one-euro"],
                    help="pose 출력에 걸 후처리 필터. none=끄기(기본), "
                         "one-euro=One Euro filter. FoundationPose 내부 추적 "
                         "상태에는 영향 없음 -- 표시/로깅용 후처리다. register "
                         "직후(새 앵커)엔 자동으로 리셋된다")
    ap.add_argument("--no-filter-pos", action="store_true",
                    help="--filter-type이 켜져 있어도 위치(x,y,z) 채널은 "
                         "필터링하지 않고 원값 그대로 둔다")
    ap.add_argument("--no-filter-rot", action="store_true",
                    help="--filter-type이 켜져 있어도 회전(쿼터니언) 채널은 "
                         "필터링하지 않고 원값 그대로 둔다")
    ap.add_argument("--one-euro-pos-mincutoff", type=float, default=1.0,
                    help="[one-euro] 위치 채널 mincutoff(Hz). 낮출수록 정지 시 "
                         "더 세게 스무딩(반응은 느려짐)")
    ap.add_argument("--one-euro-pos-beta", type=float, default=0.0,
                    help="[one-euro] 위치 채널 beta. 높일수록 빠른 움직임을 "
                         "덜 지체하며 따라간다")
    ap.add_argument("--one-euro-rot-mincutoff", type=float, default=1.0,
                    help="[one-euro] 회전 채널(쿼터니언) mincutoff(Hz)")
    ap.add_argument("--one-euro-rot-beta", type=float, default=0.0,
                    help="[one-euro] 회전 채널(쿼터니언) beta")
    ap.add_argument("--one-euro-dcutoff", type=float, default=1.0,
                    help="[one-euro] 속도 추정 자체에 쓰이는 cutoff. 보통 "
                         "기본값(1.0)에서 안 건드림")
    ap.add_argument("--pose-log", default=None,
                    help="pose를 JSON Lines로 기록할 파일 경로")
    ap.add_argument("--snapshot-dir", default="./fp_snapshots",
                    help="'s' 키로 저장할 스냅샷 폴더")
    ap.add_argument("--no-window", action="store_true",
                    help="GUI 없이 돌린다(헤드리스). ROI 대신 --init-roi 필요")
    ap.add_argument("--init-roi", default=None,
                    help="헤드리스용 초기 ROI 'x,y,w,h' (원본 해상도 기준 아님, "
                         "--scale 적용 후 기준)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="이만큼 처리하고 종료 (0이면 무제한)")
    ap.add_argument("--debug", type=int, default=0,
                    help="2 이상은 중간 렌더 결과를 붙들고 있어 VRAM을 더 쓴다")
    ap.add_argument("--debug-dir", default="./fp_debug")
    args = ap.parse_args()

    if args.no_window and not args.init_roi:
        ap.error("--no-window 를 쓰면 --init-roi 'x,y,w,h' 가 필요하다")

    cam = D455(args.width, args.height, args.fps,
               filter_depth=not args.no_depth_filter,
               bag=args.bag, repeat_bag=args.repeat_bag,
               exposure=args.exposure, white_balance=args.white_balance,
               laser_power=args.laser_power, temporal_alpha=args.temporal_alpha)
    print(f"[cam] {cam.width}x{cam.height}, depth_scale={cam.depth_scale}")
    print(f"[cam] K =\n{cam.K}")

    engine = PoseEngine(args)
    logger = PoseLog(args.pose_log) if args.pose_log else None
    os.makedirs(args.snapshot_dir, exist_ok=True)

    pose_filter = None
    if args.filter_type == "one-euro":
        if args.no_filter_pos and args.no_filter_rot:
            print("[main] !! --no-filter-pos와 --no-filter-rot을 같이 주면 "
                  "필터가 아무 채널도 안 건드림 (--filter-type none과 동일)")
        pose_filter = PoseOneEuroFilter(
            pos_mincutoff=args.one_euro_pos_mincutoff, pos_beta=args.one_euro_pos_beta,
            rot_mincutoff=args.one_euro_rot_mincutoff, rot_beta=args.one_euro_rot_beta,
            dcutoff=args.one_euro_dcutoff,
            filter_pos=not args.no_filter_pos, filter_rot=not args.no_filter_rot)

    frame_id = 0
    need_init = True
    bbox, to_origin = engine.bbox, engine.to_origin
    fps_ema = None
    t_prev = time.time()
    win = "FoundationPose (local)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        disp_w = int(round(cam.width * args.scale))
        disp_h = int(round(cam.height * args.scale))
        cv2.resizeWindow(win, disp_w, disp_h)

    try:
        while True:
            color_raw, depth_raw = cam.read()
            if color_raw is None:
                continue
            color, depth, K = downscale(color_raw, depth_raw, cam.K, args.scale)

            # ---------------------------------------------------- 초기화/추적
            if need_init:
                if args.init_roi:
                    roi = [int(v) for v in args.init_roi.split(",")]
                    mask = mask_from_box(color.shape, depth, cam.depth_scale,
                                         roi, refine=not args.no_mask_refine)
                else:
                    mask = make_mask_from_roi(
                        color, depth, cam.depth_scale,
                        refine=not args.no_mask_refine)
                if mask is None:
                    print("[main] ROI가 비어 있음. 다시 시도.")
                    continue
                pose = engine.register(color, depth, cam.depth_scale, K, mask)
                if pose is None:
                    if args.no_window:
                        break               # 헤드리스에서는 무한 재시도 금지
                    continue
                need_init = False
                if pose_filter is not None:
                    pose_filter.reset()
                print(f"[main] register 완료 ({engine.last_infer_ms:.0f} ms)")
            else:
                pose = engine.track(color, depth, cam.depth_scale, K)
                if pose is None:
                    print("[main] 추적 실패 -> 재초기화")
                    need_init = True
                    if args.no_window:
                        break
                    continue

            if pose_filter is not None:
                # register 프레임 직후엔 reset() 때문에 원값 그대로 통과되고,
                # 이후 track 프레임부터 실제로 스무딩된다.
                pose = pose_filter.filter(pose, time.time())

            if logger is not None:
                logger.write(frame_id, pose, engine.last_infer_ms)

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            # ---------------------------------------------------- 표시
            if not args.no_window:
                if args.vis_mode in ("mesh", "both"):
                    vis = engine.render_overlay(color, K, pose,
                                                alpha=args.overlay_alpha)
                else:
                    vis = color.copy()
                center_pose = pose @ np.linalg.inv(to_origin)
                if args.vis_mode in ("box", "both"):
                    draw_posed_3d_box(vis, K, center_pose, bbox)
                draw_xyz_axis(vis, K, center_pose, scale=0.06)
                t = pose[:3, 3]
                rpy = mat_to_euler_deg(pose[:3, :3])
                cv2.putText(vis, f"{fps_ema:4.1f} fps | infer "
                                 f"{engine.last_infer_ms:5.1f} ms",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, f"t   = [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, f"rpy = [{rpy[0]:+6.1f} {rpy[1]:+6.1f} {rpy[2]:+6.1f}] deg",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(win, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    need_init = True
                    engine.registered = False
                    print("[main] 재초기화 요청")
                if key == ord("s"):
                    stem = os.path.join(args.snapshot_dir, f"{frame_id:06d}")
                    cv2.imwrite(stem + "_vis.png", vis)
                    cv2.imwrite(stem + "_color.png", color)
                    cv2.imwrite(stem + "_depth.png", depth)
                    np.savetxt(stem + "_pose.txt", pose, fmt="%.6f")
                    print(f"[main] 스냅샷 저장: {stem}_*")
            elif frame_id % 30 == 0:
                t = pose[:3, 3]
                print(f"[main] #{frame_id} t=[{t[0]:+.3f} {t[1]:+.3f} "
                      f"{t[2]:+.3f}] m, {fps_ema:.1f} fps")

            frame_id += 1
            if frame_id == 30:
                engine.track_peak_report()
            if args.max_frames and frame_id >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[main] 중단")
    finally:
        cam.close()
        if logger is not None:
            logger.close()
        cv2.destroyAllWindows()
        print(f"[main] 종료. 처리 프레임 {frame_id}")


if __name__ == "__main__":
    main()

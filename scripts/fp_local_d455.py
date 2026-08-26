"""
fp_local_d455.py — 카메라와 FoundationPose가 **같은 컴퓨터**에 있을 때 쓰는 단일 프로세스 버전.

receiver_server.py(+ sender_d455.py)의 동작을 그대로 유지하되 ZeroMQ / JPEG·PNG
인코딩 계층을 전부 걷어냈다. 프레임은 numpy 배열 그대로 추정기에 들어간다.

하는 일
  1. RealSense D455에서 컬러/뎁스를 받아 컬러 프레임에 align
  2. 첫 프레임에서 마우스로 ROI를 끌어 마스크를 만들고 est.register()
  3. 이후 매 프레임 est.track_one()
  4. 4x4 pose로 3D 박스 + 좌표축을 그려 화면에 표시 (원하면 파일로 기록)

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
"""

import argparse
import json
import logging
import os
import sys
import time

# torch가 import되기 *전에* 설정해야 효과가 있다. 캐싱 할당자가 큰 블록을
# 잘게 쪼개 쥐고 있는 파편화를 줄여, nvidia-smi에 찍히는 reserved 값을 낮춘다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import pyrealsense2 as rs


LOGGER_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}


def vram(tag, reset_peak=False):
    """현재/최대 VRAM 사용량을 찍는다.

    alloc     : 실제로 텐서가 쓰고 있는 양
    reserved  : PyTorch 캐싱 할당자가 CUDA에서 받아 쥐고 있는 양.
                nvidia-smi에 보이는 값에 가깝다. alloc과 차이가 크면 파편화다.
    peak      : reset 이후 최고점. 어느 단계가 피크를 만드는지 여기서 판별한다.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
    except ImportError:
        return
    a = torch.cuda.memory_allocated() / 2 ** 20
    r = torch.cuda.memory_reserved() / 2 ** 20
    m = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"[vram] {tag:<24s} alloc {a:7.0f} MB | reserved {r:7.0f} MB "
          f"| peak {m:7.0f} MB")
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


def load_foundationpose(fp_root):
    """FoundationPose 저장소를 import path에 넣고 필요한 심볼을 가져온다."""
    if fp_root:
        sys.path.insert(0, os.path.abspath(fp_root))
    # estimater.py 가 저장소 루트에 있다.
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor  # pyright: ignore[reportMissingImports]
    import nvdiffrast.torch as dr
    return FoundationPose, ScorePredictor, PoseRefinePredictor, dr


class PoseEngine:
    """FoundationPose 래퍼. 원래 receiver_server.PoseServer에서 네트워크만 뺀 것."""

    def __init__(self, args):
        import trimesh

        self.probe = args.vram_probe
        if self.probe:
            vram("프로세스 시작", reset_peak=True)

        FoundationPose, ScorePredictor, PoseRefinePredictor, dr = load_foundationpose(args.fp_root)
        logging.getLogger().setLevel(LOGGER_LEVEL_MAP[args.log_level])

        mesh = trimesh.load(args.mesh, force="mesh")
        if args.mesh_unit == "mm":
            mesh.apply_scale(0.001)
        elif args.mesh_unit == "cm":
            mesh.apply_scale(0.01)
        ext = np.asarray(mesh.extents, dtype=float)
        print(f"[fp] mesh 로드: {args.mesh}, verts={len(mesh.vertices)}")
        print(f"[fp] mesh 실제 크기 = {ext[0]*100:.1f} x {ext[1]*100:.1f} "
              f"x {ext[2]*100:.1f} cm  <-- 실물 치수와 반드시 비교할 것")
        if ext.max() > 2.0 or ext.max() < 0.005:
            print(f"[fp] !! 경고: mesh 크기가 비현실적이다. "
                  f"--mesh-unit ({args.mesh_unit}) 이 틀렸을 가능성이 크다.")

        # crop window 계산에 쓰이는 지름. 이게 너무 작으면 FoundationPose 내부
        # compute_crop_window_tf_batch()에서 crop 폭이 0 px로 반올림되고,
        # 1/0 = inf 가 들어간 행렬을 inverse()하다 LinAlgError로 죽는다.
        self.mesh_diameter = float(np.linalg.norm(ext))
        print(f"[fp] mesh 지름(대각) = {self.mesh_diameter*100:.2f} cm")

        # 3D 박스를 그릴 때 쓰는 값. 원본 run_demo.py와 동일.
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.to_origin = to_origin
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        os.makedirs(args.debug_dir, exist_ok=True)
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=args.debug_dir,
            debug=args.debug,
            glctx=dr.RasterizeCudaContext(),
        )
        print("[fp] FoundationPose 초기화 완료")
        if self.probe:
            vram("가중치 로드 후")

        self.est_refine_iter = args.est_refine_iter
        self.track_refine_iter = args.track_refine_iter
        self.zfar = args.zfar
        self.register_chunk = args.register_chunk
        self.registered = False
        self.last_infer_ms = 0.0
        self.last_score = None   # register()가 고른 pose의 scorer 점수. track_one()은 점수를 안 낸다

        # cuDNN 알고리즘 자동 탐색. 같은 입력 크기가 반복되므로 속도에는
        # 이득이지만, 탐색 중 여러 알고리즘의 workspace를 번갈아 할당해
        # **피크 메모리를 올린다**. VRAM이 빠듯하면 --no-cudnn-benchmark.
        import torch
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

        self.set_pose_grid(args.view_subdiv, args.inplane_step)
        if not args.no_warmup:
            self.warmup()

    # -------------------------------------------------------------- 초기화 속도

    def set_pose_grid(self, subdiv, inplane_step):
        """register()가 시도할 초기 자세 가설 개수를 조절한다.

        FoundationPose 기본값은 icosphere 42뷰 x 인플레인 6단계 = 252 가설이고,
        register 시간과 VRAM이 모두 이 개수에 거의 정비례한다. 원본
        `make_rotation_grid(min_n_views=..)` 는 icosphere 세분화가 1부터
        시작해서 42뷰 밑으로 못 내려간다. 그래서 세분화 단계를 직접 강제한다.
            subdiv 0 -> 12뷰,  1 -> 42뷰(기본),  2 -> 162뷰
        """
        import estimater as fp_mod  # pyright: ignore[reportMissingImports]

        if subdiv is not None:
            orig = fp_mod.sample_views_icosphere
            fp_mod.sample_views_icosphere = (
                lambda n_views, subdivisions=None, radius=1:
                orig(n_views, subdivisions=subdiv, radius=radius))
            try:
                self.est.make_rotation_grid(min_n_views=40,
                                            inplane_step=inplane_step)
            finally:
                fp_mod.sample_views_icosphere = orig
        elif inplane_step != 60:
            self.est.make_rotation_grid(min_n_views=40,
                                        inplane_step=inplane_step)

        n = len(self.est.rot_grid)
        print(f"[fp] 초기 자세 가설 {n}개 "
              f"(기본 252 대비 {252 / max(n, 1):.1f}배 빠름/가벼움)")

    def warmup(self):
        """더미 프레임으로 register를 한 번 돌려 일회성 비용을 미리 치른다.

        첫 register가 유독 느린 이유는 알고리즘이 아니라 CUDA 컨텍스트 생성,
        cuDNN 알고리즘 탐색, 가중치 GPU 전송, nvdiffrast 컴파일 같은
        일회성 비용이다. 사용자가 ROI를 찍기 전에 여기서 끝내둔다.

        주의: warmup은 실제 register와 같은 크기의 배치를 만들므로 VRAM 피크도
        여기서 그대로 찍힌다. "시작하자마자 몇 GB"로 보이는 건 대개 이것이다.
        """
        fx = 600.0
        H, W = 480, 848
        z = max(0.3, self.mesh_diameter * fx / 120.0)   # 화면에서 ~120 px가 되게
        K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float64)
        r = max(4, int(self.mesh_diameter * fx / z / 2))
        cy, cx = H // 2, W // 2
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)

        rgb = np.random.default_rng(0).integers(
            60, 200, (H, W, 3), dtype=np.uint8)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[y0:y1, x0:x1] = z
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True

        print("[fp] warmup 중... (첫 register 비용을 미리 지불)")
        t0 = time.time()
        try:
            # 실제 register와 **같은 경로**로 돌려야 warmup이 피크를 대표한다.
            # 청크를 쓰면 warmup도 청크로.
            self._register_impl(K=K, rgb=rgb, depth=depth, mask=mask)
            print(f"[fp] warmup 완료 ({time.time() - t0:.1f}s). "
                  "실제 register는 이보다 훨씬 빠를 것")
        except Exception as e:
            print(f"[fp] warmup 실패(무시하고 계속): {type(e).__name__}: {e}")
        finally:
            self.registered = False
            if self.probe:
                vram("warmup 피크")
            self.free()
            if self.probe:
                vram("warmup + empty_cache", reset_peak=True)

    # -------------------------------------------------------------- VRAM 절약

    @staticmethod
    def free():
        """캐싱 할당자가 쥔 미사용 블록을 CUDA에 돌려준다.

        register 피크가 지나간 뒤 한 번 부르면 nvidia-smi 수치가 내려간다.
        추적 중 매 프레임 부르면 안 된다 -- 재할당 비용으로 fps가 떨어진다.
        """
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _register_impl(self, K, rgb, depth, mask):
        """register 본체. --register-chunk 가 있으면 계층적으로 나눠 돈다.

        FoundationPose의 register()는 rot_grid의 **모든** 자세 가설을 한 번에
        refiner/scorer에 밀어 넣는다. 특히 scorer(predict_score.py)는
        `bs = pose_data.rgbAs.shape[0]` 로 청크 없이 전량을 한 배치로 처리하고,
        점수 매기는 트랜스포머가 가설끼리 cross-attention(`att_cross(x, x, x)`)
        까지 하므로 메모리가 가설 개수에 대해 가파르게 는다.

        여기서는 rot_grid를 조각내 여러 번 register를 돌리고, 각 조각의 승자
        회전만 모아 마지막에 한 번 더 register("결선 라운드")를 돌려 최종
        승자를 고른다.

        주의: 조각 안에서만 순위를 매기므로 전량 비교와 결과가 100% 같지는
        않다. 결선 라운드가 그 차이를 대부분 메운다. 정확도가 최우선이면
        --register-chunk 를 끄고 --view-subdiv 로 가설 자체를 줄이는 편이 낫다.
        """
        import torch

        full = self.est.rot_grid
        chunk = int(self.register_chunk or 0)
        if chunk <= 0 or chunk >= len(full):
            return self.est.register(K=K, rgb=rgb, depth=depth,
                                     ob_mask=mask.astype(bool),
                                     iteration=self.est_refine_iter)

        winners = []
        try:
            for i in range(0, len(full), chunk):
                self.est.rot_grid = full[i:i + chunk]
                p = self.est.register(K=K, rgb=rgb, depth=depth,
                                      ob_mask=mask.astype(bool),
                                      iteration=self.est_refine_iter)
                winners.append(torch.as_tensor(np.asarray(p).reshape(4, 4),
                                               device=full.device,
                                               dtype=full.dtype))
                self.free()

            # 결선: 각 조각 승자의 **회전만** 모은다. register()가 평행이동은
            # 마스크 중심 깊이에서 다시 계산하므로 rot_grid와 같은 형식이 된다.
            grid = torch.stack(winners, dim=0).clone()
            grid[:, :3, 3] = 0.0
            self.est.rot_grid = grid
            print(f"[fp] 결선 라운드: 조각 승자 {len(winners)}개 비교")
            return self.est.register(K=K, rgb=rgb, depth=depth,
                                     ob_mask=mask.astype(bool),
                                     iteration=self.est_refine_iter)
        finally:
            self.est.rot_grid = full

    # -------------------------------------------------------------- 전처리

    def prepare(self, color_bgr, depth_u16, depth_scale):
        """카메라 원본을 FoundationPose 입력 형식으로 변환.

        원격 버전과 달리 JPEG/PNG 왕복이 없다. 컬러는 무손실 그대로 들어가고
        뎁스도 재인코딩 없이 그대로 쓰인다.
        """
        # FoundationPose는 RGB를 기대한다. RealSense bgr8 출력은 BGR.
        rgb = color_bgr[..., ::-1].copy()

        # 뎁스는 미터 단위 float32. 유효 범위 밖은 0으로 눌러야 한다.
        depth = depth_u16.astype(np.float32) * float(depth_scale)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0.0
        return rgb, depth

    # -------------------------------------------------------------- 진단

    def diagnose(self, mask, depth, K=None):
        """register 직전에 입력 품질을 찍는다. 추적이 이상할 때 여기부터 본다."""
        n_mask = int(mask.sum())
        if n_mask == 0:
            print("[fp] !! 마스크가 완전히 비었다")
            return
        d = depth[mask]
        d_valid = d[d > 0]
        ratio = len(d_valid) / max(n_mask, 1)
        ys, xs = np.where(mask)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

        print(f"[fp] 마스크 {n_mask} px, 화면상 크기 {bw}x{bh} px, "
              f"유효 뎁스 {ratio*100:.0f}%", end="")
        if len(d_valid) > 0:
            med = float(np.median(d_valid))
            print(f", 거리 중앙값 {med:.3f} m")
            if med < 0.5:
                print("[fp] !! 경고: 물체가 D455 min-Z(~0.5 m)보다 가깝다. "
                      "뎁스가 신뢰할 수 없다. 더 멀리 두거나 D405를 쓸 것")
            if K is not None:
                # FoundationPose가 잡을 crop 창의 반경(px). 0.5 px 미만이면
                # 내부에서 폭이 0으로 반올림돼 inverse()가 터진다.
                r_px = self.mesh_diameter * 1.2 / 2 * float(K[0, 0]) / med
                print(f"[fp] 예상 crop 반경 {r_px:.1f} px")
                if r_px < 2.0:
                    print("[fp] !! 치명적: crop 창이 사실상 0 px다. "
                          "mesh가 실물보다 훨씬 작게 스케일됐다 "
                          "(--mesh-unit 확인). 이대로면 linalg.inv 오류로 죽는다")
        else:
            print("\n[fp] !! 경고: 마스크 안에 유효한 뎁스가 하나도 없다")

        if min(bw, bh) < 30:
            print(f"[fp] !! 경고: 물체가 화면에서 너무 작다({bw}x{bh} px). "
                  "FoundationPose는 내부적으로 crop 후 리사이즈하므로 "
                  "짧은 변이 최소 40~60 px는 되어야 한다")
        if ratio < 0.5:
            print("[fp] !! 경고: 뎁스 구멍이 많다. 표면 광택/반투명 또는 "
                  "min-Z 미만 거리 의심")

    @staticmethod
    def _explain(exc, stage):
        """FoundationPose 내부 예외를 사람이 읽을 수 있는 원인으로 번역."""
        import traceback
        traceback.print_exc()
        s = f"{type(exc).__name__}: {exc}"
        if "singular" in str(exc) or "LinAlg" in type(exc).__name__:
            if stage == "register":
                s += (" | crop 창 폭이 0 px로 반올림된 것. mesh가 실물보다 "
                      "너무 작게 스케일됐다 -- --mesh-unit 을 확인할 것")
            else:
                s += (" | 직전 자세가 발산(z<=0 또는 NaN)했다. 재초기화 필요")
        if "out of memory" in str(exc).lower():
            s += (" | VRAM 부족. --view-subdiv 0 --inplane-step 120 으로 가설을 "
                  "줄이거나 --register-chunk 32 로 나눠 돌릴 것. "
                  "--vram-probe 로 어느 단계가 피크인지 먼저 확인")
        return s

    # -------------------------------------------------------------- 공개 API

    def register(self, color_bgr, depth_u16, depth_scale, K, mask):
        """초기 자세 추정. 성공하면 4x4 pose, 실패하면 None."""
        if mask is None or mask.sum() < 50:
            print("[fp] init 실패: 마스크가 비어 있음")
            return None
        rgb, depth = self.prepare(color_bgr, depth_u16, depth_scale)
        self.diagnose(mask.astype(bool), depth, K)

        t0 = time.time()
        try:
            pose = self._register_impl(K=K, rgb=rgb, depth=depth, mask=mask)
        except Exception as e:
            print(f"[fp] register 실패: {self._explain(e, 'register')}")
            self.registered = False
            self.free()
            return None
        self.last_infer_ms = (time.time() - t0) * 1000.0

        # register()가 내부에서 뽑은 가설들 중 최고 점수. self.est.scores는
        # scorer.predict() 결과를 내림차순 정렬해 둔 것 -- [0]이 채택된 pose의 점수.
        scores = getattr(self.est, "scores", None)
        self.last_score = float(scores[0]) if scores is not None and len(scores) else None

        # register 피크가 지나갔다. 추적 중에는 가설이 1개뿐이라 훨씬 적게 쓴다.
        if self.probe:
            vram("register 피크")
        self.free()
        if self.probe:
            vram("register + empty_cache", reset_peak=True)

        pose = self._validate(pose)
        self.registered = pose is not None
        return pose

    def track(self, color_bgr, depth_u16, depth_scale, K):
        """이후 프레임 추적. 실패하면 None을 주고 registered를 내린다."""
        if not self.registered:
            return None
        rgb, depth = self.prepare(color_bgr, depth_u16, depth_scale)

        t0 = time.time()
        try:
            pose = self.est.track_one(rgb=rgb, depth=depth, K=K,
                                      iteration=self.track_refine_iter)
        except Exception as e:
            # 직전 자세가 카메라 뒤로 넘어갔거나 NaN이 된 상태. 추적을 버린다.
            print(f"[fp] track 실패: {self._explain(e, 'track')}")
            self.registered = False
            self.free()
            return None
        self.last_infer_ms = (time.time() - t0) * 1000.0

        pose = self._validate(pose)
        if pose is None:
            self.registered = False
        return pose

    def track_peak_report(self):
        """추적 정상상태의 VRAM. register 피크와 비교하면 차이가 크게 난다."""
        if self.probe:
            vram("추적 정상상태")

    def _validate(self, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            print("[fp] pose에 NaN - 추적 실패")
            return None
        return pose


# ==================================================================== 카메라


class D455:
    """align된 컬러/뎁스 프레임과 컬러 intrinsics를 내주는 얇은 래퍼.

    원격 버전과 다른 점: 추론이 캡처보다 느리므로 매 루프마다 큐에 쌓인
    프레임을 전부 비우고 **가장 최근 것만** 쓴다. 이걸 안 하면 화면이
    실제보다 몇 프레임 뒤처져 보인다.
    """

    def __init__(self, width=848, height=480, fps=30, filter_depth=True,
                 bag=None, repeat_bag=False, depth_min=0.2, depth_max=4.0):
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

            # mode 2 = "Nearest from around": 구멍을 배경(먼 값)이 아니라 전경
            # 쪽 값으로 채운다. 기본 mode 1("Farest from around")은 광택/검은
            # 표면 때문에 물체 위에 뚫린 구멍에 배경 depth가 새어 들어오기 쉽다.
            self._hole = rs.hole_filling_filter(2)

        if not bag:
            # 자동 노출이 안정될 때까지 몇 프레임 버린다.
            for _ in range(30):
                self.pipeline.wait_for_frames()

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


# ==================================================================== 마스크


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


# ==================================================================== 시각화


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


# ==================================================================== pose 기록


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
        self.f = open(path, "w", encoding="utf-8")
        print(f"[log] pose 기록: {path}")

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


# ==================================================================== main


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

    # 카메라를 먼저 연다. 모델 로딩(수십 초)에 시간을 쓴 뒤에야
    # "카메라가 안 잡힌다"는 걸 알게 되는 상황을 피한다.
    cam = D455(args.width, args.height, args.fps,
               filter_depth=not args.no_depth_filter,
               bag=args.bag, repeat_bag=args.repeat_bag)
    print(f"[cam] {cam.width}x{cam.height}, depth_scale={cam.depth_scale}")
    print(f"[cam] K =\n{cam.K}")

    engine = PoseEngine(args)
    logger = PoseLog(args.pose_log) if args.pose_log else None
    os.makedirs(args.snapshot_dir, exist_ok=True)

    frame_id = 0
    need_init = True
    bbox, to_origin = engine.bbox, engine.to_origin
    fps_ema = None
    t_prev = time.time()
    win = "FoundationPose (local)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # WINDOW_NORMAL은 첫 imshow 전까지 임의의(보통 작은) 기본 크기로 뜬다.
        # 실제 표시될 프레임 해상도(캡처 해상도 x --scale)로 미리 맞춰
        # 마우스로 테두리를 끌지 않아도 되게 한다.
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
                print(f"[main] register 완료 ({engine.last_infer_ms:.0f} ms)")
            else:
                pose = engine.track(color, depth, cam.depth_scale, K)
                if pose is None:
                    print("[main] 추적 실패 -> 재초기화")
                    need_init = True
                    if args.no_window:
                        break
                    continue

            if logger is not None:
                logger.write(frame_id, pose, engine.last_infer_ms)

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            # ---------------------------------------------------- 표시
            if not args.no_window:
                vis = color.copy()
                center_pose = pose @ np.linalg.inv(to_origin)
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
                # # track_one()은 점수를 계산하지 않는다. register() 때 고른 pose의
                # # scorer 점수를 그대로 들고 있는 것 -- 매 프레임 갱신되지 않는다.
                # score_txt = (f"{engine.last_score:+.3f}" if engine.last_score is not None else "N/A")
                # cv2.putText(vis, f"score(register) = {score_txt}",
                #             (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             (0, 255, 255), 2, cv2.LINE_AA)
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

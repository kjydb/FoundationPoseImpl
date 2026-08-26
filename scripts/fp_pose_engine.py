"""FoundationPose 래퍼: mesh 로드, register/track, mesh 오버레이 렌더링."""

import logging
import os
import sys
import time

import cv2
import numpy as np

from fp_vram import vram

LOGGER_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}


def load_foundationpose(fp_root):
    """FoundationPose 저장소를 import path에 넣고 필요한 심볼을 가져온다."""
    if fp_root:
        sys.path.insert(0, os.path.abspath(fp_root))
    # estimater.py 가 저장소 루트에 있다.
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor  # pyright: ignore[reportMissingImports]
    import nvdiffrast.torch as dr
    import Utils as fp_utils  # pyright: ignore[reportMissingImports]  # nvdiffrast_render() 등 시각화 유틸
    return FoundationPose, ScorePredictor, PoseRefinePredictor, dr, fp_utils


def make_symmetry_tfs(axis, order):
    """mesh 로컬 좌표계에서 axis 둘레 order-겹 회전대칭 변환들.

    물체가 실제로 이 대칭을 가지는데 안 알려주면, register()의 회전 가설들이
    "겉보기엔 똑같은" 서로 다른 회전 사이를 프레임마다 오가며 회전만 흔들리는
    노이즈로 나타난다. 여기서 만든 변환들을 FoundationPose(symmetry_tfs=...)에
    넘기면 rot_grid 클러스터링(mycpp.cluster_poses)이 그 대칭 후보들을 하나로
    묶어줘서 흔들림이 줄어든다.
    """
    import trimesh
    vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    tfs = [trimesh.transformations.rotation_matrix(2 * np.pi * k / order, vec)
          for k in range(order)]
    return np.stack(tfs, axis=0)


class PoseEngine:
    """FoundationPose 래퍼. 원래 receiver_server.PoseServer에서 네트워크만 뺀 것."""

    def __init__(self, args):
        import trimesh

        self.probe = args.vram_probe
        if self.probe:
            vram("프로세스 시작", reset_peak=True)

        FoundationPose, ScorePredictor, PoseRefinePredictor, dr, fp_utils = load_foundationpose(args.fp_root)
        self.fp_utils = fp_utils
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

        symmetry_tfs = None
        if args.symmetry_axis:
            symmetry_tfs = make_symmetry_tfs(args.symmetry_axis, args.symmetry_order)
            print(f"[fp] 대칭 등록: {args.symmetry_axis}축 {args.symmetry_order}겹 "
                  f"({360 / args.symmetry_order:.0f}도 간격)")

        os.makedirs(args.debug_dir, exist_ok=True)
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            symmetry_tfs=symmetry_tfs,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=args.debug_dir,
            debug=args.debug,
            glctx=dr.RasterizeCudaContext(),
        )
        print("[fp] FoundationPose 초기화 완료")
        if self.probe:
            vram("가중치 로드 후")

        # register()/track_one()이 돌려주는 pose는 FoundationPose가 내부적으로
        # bbox 중심으로 재원점화한 self.est.mesh가 아니라 **원본 mesh 좌표계**
        # 기준이다 (estimater.py의 reset_object에서 mesh를 -model_center만큼
        # 옮기고, register/track_one 리턴 직전에 get_tf_to_centered_mesh()로
        # 그 이동을 다시 상쇄해서 내보낸다). 오버레이 렌더링도 pose와 같은
        # 원본 좌표계를 써야 해서, 재원점화 전 원본을 담고 있는 self.est.mesh_ori로
        # mesh_tensors를 따로 만들어 둔다.
        self.render_mesh = self.est.mesh_ori
        self.render_mesh_tensors = self.fp_utils.make_mesh_tensors(self.render_mesh)

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

    # -------------------------------------------------------------- mesh 오버레이

    def render_overlay(self, color_bgr, K, pose, alpha=0.6, outline=True):
        """추정된 pose로 실제 mesh를 렌더링해 컬러 프레임 위에 반투명하게 씌운다.

        register/track이 내부적으로 refiner/scorer용으로 쓰는 것과 같은
        glctx(nvdiffrast CUDA 컨텍스트)·mesh_tensors를 재사용하므로 컨텍스트를
        새로 만들지 않는다. depth==0인 픽셀이 배경이라 그걸로 실루엣 마스크를
        만든다 (nvdiffrast_render가 배경을 0으로 밀어둔다).
        """
        import torch

        H, W = color_bgr.shape[:2]
        ob_in_cam = torch.as_tensor(pose.reshape(1, 4, 4), device='cuda',
                                    dtype=torch.float)
        rendered, depth, _ = self.fp_utils.nvdiffrast_render(
            K=K, H=H, W=W, ob_in_cams=ob_in_cam, glctx=self.est.glctx,
            mesh_tensors=self.render_mesh_tensors, mesh=self.render_mesh,
            output_size=(H, W), use_light=True)

        rendered = rendered[0].detach().cpu().numpy()  # (H, W, 3) RGB, 0..1
        mask = (depth[0].detach().cpu().numpy() > 1e-6)
        if not mask.any():
            return color_bgr

        rendered_bgr = np.clip(rendered[..., ::-1] * 255.0, 0, 255)
        vis = color_bgr.copy().astype(np.float32)
        vis[mask] = alpha * rendered_bgr[mask] + (1 - alpha) * vis[mask]
        vis = vis.astype(np.uint8)

        if outline:
            m8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

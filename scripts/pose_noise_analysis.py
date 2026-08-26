"""
pose_noise_analysis.py — fp_local_d455.py --pose-log 로 기록한 JSON Lines를
오프라인으로 분석해서 pose 추정의 "측정 노이즈"를 정량화한다.

OBB가 가만히 있는 물체에서도 흔들리는 이유를 조사하려면, 그게 실제 추적
불안정(노이즈)인지 실제 물체가 움직인 것인지부터 구분해야 한다. 이 스크립트는
로그를 프레임별 속도(위치/회전)로 자동 분류해 두 구간을 나눈다.

  - 정지 구간 (물체가 안 움직인다고 판단되는 구간)
      노이즈 = 구간 평균 pose 대비 각 프레임의 편차의 표준편차.
      "진짜" 표준편차 그대로다 -- 물체가 안 움직였으니 편차는 전부 노이즈.
  - 움직임 구간 (물체가 실제로 움직이는 구간)
      노이즈 = 이동평균으로 만든 "저역통과(추세)" pose 대비 각 프레임의 잔차의
      표준편차. 저주파(실제 움직임)를 스무딩으로 빼고 남은 고주파 성분을
      노이즈로 본다.

구간 분류는 이동평균으로 스무딩한 위치/회전 궤적의 프레임간 속도를 임계값과
비교해서 정한다 (--pos-speed-thresh, --rot-speed-thresh). 너무 짧은 구간
(--min-segment-frames 미만)은 전환 구간으로 보고 통계에서 뺀다.

먼저 기록
    python fp_local_d455.py --mesh ... --mesh-unit mm --fp-root ... \
        --pose-log ./fp_pose.jsonl

그 다음 분석 (FoundationPose/torch 환경 필요 없음, numpy만 있으면 됨)
    python pose_noise_analysis.py --log ./fp_pose.jsonl
    python pose_noise_analysis.py --log ./fp_pose.jsonl --plot --plot-out noise.png

주의: 정지 구간을 하나도 못 찾으면 --pos-speed-thresh / --rot-speed-thresh를
올리거나, 애초에 물체를 몇 초간 완전히 가만히 두고 녹화한 로그를 써야 한다.
"""

import argparse
import json

import numpy as np


# ==================================================================== 로드


def load_log(path):
    frame_id, stamp, t, q = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame_id.append(int(rec["frame_id"]))
            stamp.append(float(rec["stamp"]))
            t.append(rec["t"])
            q.append(rec["q_wxyz"])
    if len(frame_id) < 3:
        raise ValueError(f"{path}: 유효한 레코드가 너무 적다 ({len(frame_id)}개)")
    return (np.asarray(frame_id, dtype=np.int64),
            np.asarray(stamp, dtype=np.float64),
            np.asarray(t, dtype=np.float64),      # (N,3) meters
            np.asarray(q, dtype=np.float64))      # (N,4) w,x,y,z


# ==================================================================== 유틸


def moving_average(x, window):
    """x: (N,) 또는 (N,D). reflect padding이라 끝쪽도 길이가 줄지 않는다."""
    if window <= 1:
        return x.astype(np.float64).copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x2 = x if x.ndim == 2 else x[:, None]
    xp = np.pad(x2, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(x2, dtype=np.float64)
    for c in range(x2.shape[1]):
        out[:, c] = np.convolve(xp[:, c], kernel, mode="valid")
    return out if x.ndim == 2 else out[:, 0]


def fix_quat_continuity(q):
    """q와 -q는 같은 회전이다. 프레임간 부호가 튀지 않게 이전 프레임과
    가까운 쪽 부호로 통일한다 (스무딩/평균 전에 반드시 필요)."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def quat_angle_deg(q1, q2):
    """단위 쿼터니언 쌍 사이의 측지 각도(deg). q1,q2: (...,4)."""
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def normalize_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ==================================================================== 구간 분류


def block_speeds(stamp, t, q_fixed, half_window):
    """전/후 half_window프레임 블록 평균의 차이로 속도를 추정한다. mm/s, deg/s.

    인접 프레임끼리 단순 미분하면 노이즈가 미분에서 오히려 증폭된다 (스무딩으로
    노이즈를 sqrt(window)만큼만 줄여도, 1프레임 baseline으로 나누는 순간 다시
    커진다). 블록 평균끼리 비교하면 baseline이 half_window프레임으로 늘어나
    노이즈가 훨씬 강하게 억제되고, 실제 저주파 움직임은 거의 그대로 남는다.
    대가로 구간 경계가 half_window프레임 정도 무뎌진다.
    """
    n = len(stamp)
    h = max(1, int(half_window))
    pos_speed = np.zeros(n)
    rot_speed = np.zeros(n)
    for i in range(n):
        i0, i1 = max(0, i - h), min(n, i + h)
        if i - i0 < 2 or i1 - i < 2:
            continue  # 가장자리는 아래에서 가까운 유효값으로 채운다
        before_pos = t[i0:i].mean(axis=0)
        after_pos = t[i:i1].mean(axis=0)
        dt = stamp[i:i1].mean() - stamp[i0:i].mean()
        if dt <= 1e-6:
            continue
        pos_speed[i] = np.linalg.norm(after_pos - before_pos) * 1000.0 / dt

        before_q = normalize_rows(q_fixed[i0:i].mean(axis=0, keepdims=True))[0]
        after_q = normalize_rows(q_fixed[i:i1].mean(axis=0, keepdims=True))[0]
        rot_speed[i] = quat_angle_deg(before_q, after_q) / dt

    # 가장자리(양끝 h프레임 근처, 계산 못 한 구간)는 가장 가까운 유효값으로 채운다.
    # 무효 구간은 처음/끝에만 붙어 있는 두 덩어리뿐이라 단순 forward/backward fill로 충분하다.
    valid = np.array([(min(n, i + h) - i >= 2) and (i - max(0, i - h) >= 2)
                      for i in range(n)])
    if valid.any():
        first, last = np.argmax(valid), n - 1 - np.argmax(valid[::-1])
        pos_speed[:first] = pos_speed[first]
        pos_speed[last + 1:] = pos_speed[last]
        rot_speed[:first] = rot_speed[first]
        rot_speed[last + 1:] = rot_speed[last]
    return pos_speed, rot_speed


def runs_from_labels(labels):
    """bool 배열 -> [(start, end_exclusive, label), ...] 연속 구간 목록."""
    runs = []
    start = 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            runs.append((start, i, bool(labels[start])))
            start = i
    return runs


# ==================================================================== 구간별 통계


def segment_stats(t_mm, q_fixed, pos_smooth_mm, quat_smooth, sl, still):
    """sl 구간의 노이즈 통계. still이면 구간 평균 대비, 아니면 스무딩 잔차."""
    ts = t_mm[sl]
    qs = q_fixed[sl]

    if still:
        ref_pos = np.repeat(ts.mean(axis=0, keepdims=True), len(ts), axis=0)
        ref_q = normalize_rows(qs.mean(axis=0, keepdims=True))
        ref_q = np.repeat(ref_q, len(qs), axis=0)
    else:
        ref_pos = pos_smooth_mm[sl]
        ref_q = quat_smooth[sl]

    resid_pos = ts - ref_pos                    # (n,3) mm
    resid_ang = quat_angle_deg(qs, ref_q)        # (n,) deg

    return {
        "n": len(ts),
        "resid_pos_mm": resid_pos,
        "resid_ang_deg": resid_ang,
        "pos_std_mm": resid_pos.std(axis=0),
        "pos_rms_mm": float(np.sqrt(np.mean(np.sum(resid_pos ** 2, axis=1)))),
        "rot_std_deg": float(resid_ang.std()),
        "rot_rms_deg": float(np.sqrt(np.mean(resid_ang ** 2))),
    }


def pooled_stats(segs):
    if not segs:
        return None
    resid_pos = np.concatenate([s["resid_pos_mm"] for s in segs], axis=0)
    resid_ang = np.concatenate([s["resid_ang_deg"] for s in segs], axis=0)
    return {
        "n_segments": len(segs),
        "n_frames": sum(s["n"] for s in segs),
        "pos_std_mm": resid_pos.std(axis=0),
        "pos_rms_mm": float(np.sqrt(np.mean(np.sum(resid_pos ** 2, axis=1)))),
        "rot_std_deg": float(resid_ang.std()),
        "rot_rms_deg": float(np.sqrt(np.mean(resid_ang ** 2))),
    }


# ==================================================================== 표시


def fmt_xyz(v):
    return f"[{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}]"


def print_report(frame_id, stamp, results, skipped, pooled_still, pooled_moving):
    print(f"[log] 프레임 {len(frame_id)}개, 구간 {len(results)}개 "
          f"(전환/짧은 구간 {len(skipped)}개는 제외)")
    print()
    header = (f"{'#':>3} {'종류':<6} {'frame':>13} {'dur(s)':>7} {'n':>5}  "
              f"{'pos_std(x,y,z) mm':<22} {'pos_rms':>7}  "
              f"{'rot_std':>7} {'rot_rms':>7}")
    print(header)
    print("-" * len(header))
    for i, (kind, sl, stats) in enumerate(results):
        f0, f1 = frame_id[sl.start], frame_id[sl.stop - 1]
        dur = stamp[sl.stop - 1] - stamp[sl.start]
        print(f"{i:>3} {kind:<6} {f0:>6d}-{f1:<6d} {dur:>7.2f} {stats['n']:>5}  "
              f"{fmt_xyz(stats['pos_std_mm']):<22} {stats['pos_rms_mm']:>7.3f}  "
              f"{stats['rot_std_deg']:>7.3f} {stats['rot_rms_deg']:>7.3f}")
    print()

    for name, pooled in (("정지(still)", pooled_still), ("움직임(moving)", pooled_moving)):
        print(f"=== {name} 구간 전체 ===")
        if pooled is None:
            print("  해당 구간 없음 (임계값을 조정하거나 로그를 다시 확인할 것)")
            continue
        print(f"  구간 {pooled['n_segments']}개, 프레임 {pooled['n_frames']}개")
        print(f"  위치 노이즈 std (x,y,z) = {fmt_xyz(pooled['pos_std_mm'])} mm, "
              f"3D RMS = {pooled['pos_rms_mm']:.3f} mm")
        print(f"  회전 노이즈 std = {pooled['rot_std_deg']:.3f} deg, "
              f"RMS = {pooled['rot_rms_deg']:.3f} deg")
        print()


def make_plot(frame_id, t_mm, pos_smooth_mm, results, out_path):
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    colors = {"still": "#4C9F70", "moving": "#E08E45"}
    labels_seen = set()

    for kind, sl, _ in results:
        for ax in axes:
            lbl = kind if kind not in labels_seen else None
            ax.axvspan(frame_id[sl.start], frame_id[sl.stop - 1],
                       color=colors[kind], alpha=0.15, label=lbl)
        labels_seen.add(kind)

    axis_names = ["x", "y", "z"]
    for i, ax in enumerate(axes[:3]):
        ax.plot(frame_id, t_mm[:, i], color="0.3", lw=0.8, label="raw")
        ax.plot(frame_id, pos_smooth_mm[:, i], color="tab:blue", lw=1.2,
                label="smoothed")
        ax.set_ylabel(f"{axis_names[i]} (mm)")
        ax.grid(alpha=0.3)

    ax = axes[3]
    for kind, sl, stats in results:
        ax.plot(frame_id[sl], stats["resid_ang_deg"], color="tab:red", lw=0.9)
    ax.set_ylabel("rot residual (deg)")
    ax.set_xlabel("frame_id")
    ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Pose noise analysis (shaded = still / moving segments)")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[plot] 저장: {out_path}")
    else:
        plt.show()


# ==================================================================== main


def main():
    ap = argparse.ArgumentParser(
        description="fp_local_d455.py --pose-log 로그의 정지/움직임 노이즈 분석")
    ap.add_argument("--log", required=True, help="--pose-log로 기록한 .jsonl 경로")
    ap.add_argument("--smooth-window", type=int, default=7,
                    help="움직임 구간에서 노이즈를 뺄 '추세' 기준선을 만드는 "
                         "이동평균 윈도(프레임, 홀수). 너무 크면 실제 움직임까지 "
                         "노이즈로 잡히고, 너무 작으면 노이즈가 추세에 섞여 들어간다")
    ap.add_argument("--classify-window", type=int, default=15,
                    help="정지/움직임 분류용 속도를 잴 때 쓰는 반쪽 윈도(프레임). "
                         "전/후 이 프레임 수만큼 블록 평균을 비교해 속도를 낸다. "
                         "--smooth-window보다 커야 노이즈에 안 흔들린다 (구간 경계는 "
                         "그만큼 무뎌짐)")
    ap.add_argument("--pos-speed-thresh", type=float, default=5.0,
                    help="이보다 빠르면(mm/s) 해당 프레임을 '움직임'으로 본다")
    ap.add_argument("--rot-speed-thresh", type=float, default=3.0,
                    help="이보다 빠르면(deg/s) 해당 프레임을 '움직임'으로 본다")
    ap.add_argument("--min-segment-frames", type=int, default=10,
                    help="이보다 짧은 구간은 전환 구간으로 보고 통계에서 뺀다")
    ap.add_argument("--plot", action="store_true", help="시계열 그래프를 그린다")
    ap.add_argument("--plot-out", default=None,
                    help="그래프를 파일로 저장 (지정 안 하면 화면에 표시)")
    args = ap.parse_args()

    frame_id, stamp, t, q = load_log(args.log)
    print(f"[log] {args.log} 로드: 프레임 {len(frame_id)}개, "
          f"{stamp[-1] - stamp[0]:.1f}s")

    q_fixed = fix_quat_continuity(q)
    pos_smooth_m = moving_average(t, args.smooth_window)
    quat_smooth = normalize_rows(moving_average(q_fixed, args.smooth_window))

    pos_speed, rot_speed = block_speeds(stamp, t, q_fixed, args.classify_window)
    moving = (pos_speed > args.pos_speed_thresh) | (rot_speed > args.rot_speed_thresh)

    t_mm = t * 1000.0
    pos_smooth_mm = pos_smooth_m * 1000.0

    results, skipped = [], []
    for start, end, is_moving in runs_from_labels(moving):
        if end - start < args.min_segment_frames:
            skipped.append((start, end))
            continue
        sl = slice(start, end)
        stats = segment_stats(t_mm, q_fixed, pos_smooth_mm, quat_smooth, sl,
                              still=not is_moving)
        results.append(("moving" if is_moving else "still", sl, stats))

    still_segs = [r[2] for r in results if r[0] == "still"]
    moving_segs = [r[2] for r in results if r[0] == "moving"]

    print_report(frame_id, stamp, results, skipped,
                 pooled_stats(still_segs), pooled_stats(moving_segs))

    if args.plot:
        make_plot(frame_id, t_mm, pos_smooth_mm, results, args.plot_out)


if __name__ == "__main__":
    main()

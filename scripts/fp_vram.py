"""GPU 메모리 사용량을 찍는 작은 유틸리티.

register()/warmup() 등 단계별로 VRAM 피크가 어디서 나오는지 판별할 때 쓴다.
fp_pose_engine.PoseEngine이 --vram-probe 옵션과 함께 사용한다.
"""


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

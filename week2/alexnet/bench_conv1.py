"""Conv1 계산량 체감: 순수 Python vs NumPy vs PyTorch CPU vs PyTorch GPU

    python bench_conv1.py
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

H, K, S, C_IN, C_OUT, OUT = 227, 11, 4, 3, 96, 55
MAC_PER_IMAGE = OUT * OUT * C_OUT * K * K * C_IN          # 55*55*96 출력 x 363 곱셈
IMAGES_PER_EPOCH, EPOCHS = 1_200_000, 90


def report(name, sec_per_image):
    # 학습은 순전파 + 역전파(약 2배) ≈ 순전파의 3배
    total = sec_per_image * 3 * IMAGES_PER_EPOCH * EPOCHS
    print(f"{name:<22} 이미지 1장 {sec_per_image * 1000:>12.3f} ms | "
          f"Conv1만 90 epoch 학습 ≈ {total / 3600:>12,.1f} 시간 ({total / 86400 / 365:,.2f} 년)")


x = np.random.randn(C_IN, H, H).astype(np.float32)
w = np.random.randn(C_OUT, C_IN, K, K).astype(np.float32)
print(f"Conv1 이미지 1장 곱셈 수: {MAC_PER_IMAGE:,}  (약 {MAC_PER_IMAGE / 1e6:.0f}M)\n")

# 1. 순수 Python: 필터 1개의 출력 한 줄(55칸)만 계산하고 전체로 환산
t = time.perf_counter()
for j in range(OUT):
    s = 0.0
    for c in range(C_IN):
        for a in range(K):
            for b in range(K):
                s += float(x[c, a, j * S + b]) * float(w[0, c, a, b])
report("순수 Python (for문)", (time.perf_counter() - t) * OUT * C_OUT)

# 2. NumPy: 11x11 영역 3025개를 펼쳐(im2col) 행렬곱 한 번
t = time.perf_counter()
patches = np.lib.stride_tricks.sliding_window_view(x, (C_IN, K, K))[0, ::S, ::S].reshape(OUT * OUT, -1)
out = patches @ w.reshape(C_OUT, -1).T
report("NumPy (행렬곱)", time.perf_counter() - t)

# 3, 4. PyTorch: 배치 128장으로 측정
xb, wb = torch.randn(128, C_IN, H, H), torch.from_numpy(w)
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    xd, wd = xb.to(dev), wb.to(dev)
    for _ in range(3):                      # 워밍업
        F.conv2d(xd, wd, stride=S)
    if dev == "cuda":
        torch.cuda.synchronize()
    t, n = time.perf_counter(), 10
    for _ in range(n):
        F.conv2d(xd, wd, stride=S)
    if dev == "cuda":
        torch.cuda.synchronize()            # GPU 는 비동기라 끝날 때까지 기다려야 정확
    name = f"PyTorch {dev.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
    report(name[:22], (time.perf_counter() - t) / n / 128)

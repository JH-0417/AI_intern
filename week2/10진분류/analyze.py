"""학습된 모델(best.pt)이 테스트셋에서 무엇을 틀렸는지 분석한다 (정성 분석).

만드는 그림 (analysis/ 폴더에 저장):
    1_confusion.png   혼동 행렬: 정답(행) → 예측(열). 어떤 클래스를 어떤 클래스로 헷갈리는지
    2_per_class.png   클래스별 정확도: 어떤 클래스가 특히 어려운지
    3_top_pairs.png   가장 많이 헷갈린 (정답 → 예측) 쌍마다 실제 틀린 사진 예시
    4_confident.png   모델이 "확신하고" 틀린 사진들 (예측 확률이 가장 높았던 오답)

사용 예:
    python analyze.py                       # runs 아래에서 best test acc 가 가장 높은 실행을 자동 선택
    python analyze.py --run runs/sweep_dropout_.../cifar10_aug_lr0.001_bs128_do0.0_seed1_...
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import CIFAR10Dataset, CIFAR_CLASSES   # ← 데이터셋마다 다른 부분
from model import SimpleCNN

# ============================================================ 데이터셋마다 다른 설정
NAME = "CIFAR-10"
CLASSES = CIFAR_CLASSES                                       # airplane, automobile, ..., truck
make_test_set = lambda: CIFAR10Dataset(train=False)           # 테스트셋 (증강 없이 원본)
make_model = lambda cfg: SimpleCNN(in_channels=3, img_size=32, dropout=cfg["dropout"])


def to_image(x):
    """정규화된 텐서 (3, 32, 32) → 화면에 그릴 수 있는 0~1 배열 (3, 32, 32). 채널별 정규화를 거꾸로 되돌림"""
    return (x * CIFAR10Dataset.STD + CIFAR10Dataset.MEAN).clamp(0, 1).numpy()


CMAP = None                                                   # 컬러 이미지라 색 그대로 표시


# ============================================================ 공통 부분
def find_best_run():
    """runs 아래 모든 history.json 중 best_test_acc 가 가장 높은 실행 폴더를 찾는다."""
    best, best_acc = None, -1
    for path in glob.glob("runs/**/history.json", recursive=True):
        with open(path, encoding="utf-8") as f:
            acc = json.load(f)["best_test_acc"]
        if acc > best_acc:
            best, best_acc = os.path.dirname(path), acc
    return best


@torch.no_grad()                                              # 분석만 하니까 그래디언트 계산 끔
def predict(model, loader, device):
    """테스트셋 전체에 대해 (정답, 예측, 예측 확률 전체)를 모아서 돌려준다."""
    ys, probs = [], []
    for x, y in loader:
        p = model(x.to(device)).softmax(1).cpu()              # 점수(logit) → 확률 (합 = 1)
        ys.append(y)
        probs.append(p)
    y, p = torch.cat(ys), torch.cat(probs)                    # 배치들을 하나로 이어 붙이기
    return y.numpy(), p.argmax(1).numpy(), p.numpy()          # 정답, 예측(가장 큰 확률 위치), 확률 (N, 10)


def plot_confusion(cm, out):
    """혼동 행렬 그림. cm[i, j] = 정답이 i 인데 j 로 예측한 개수"""
    norm = cm / cm.sum(1, keepdims=True)                      # 행마다 비율로 (정답 클래스 기준 %)
    off = norm.copy()
    np.fill_diagonal(off, np.nan)                             # 대각선(정답)은 색칠에서 빼서 "틀린 칸"이 잘 보이게
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(off, cmap="Reds")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:                                      # 0 인 칸은 숫자 생략
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="black" if i != j else "gray", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45 if len(CLASSES[0]) > 2 else 0)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{NAME} confusion matrix (count; color = error rate, diagonal = correct)")
    fig.colorbar(im, ax=ax, label="error rate (of true class)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_class(cm, out):
    """클래스별 정확도 막대그래프 (낮은 순으로 정렬)"""
    acc = cm.diagonal() / cm.sum(1)
    order = np.argsort(acc)                                   # 어려운 클래스부터
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([CLASSES[i] for i in order], acc[order] * 100)
    for b, a in zip(bars, acc[order]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{a * 100:.1f}", ha="center", va="bottom", fontsize=8)
    lo = max(0, acc.min() * 100 - 5)
    ax.set_ylim(lo, 100)                                      # 차이가 잘 보이도록 y축 아래쪽을 자름
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{NAME} per-class test accuracy (sorted, overall {cm.diagonal().sum() / cm.sum() * 100:.2f}%)")
    ax.tick_params(axis="x", rotation=45 if len(CLASSES[0]) > 2 else 0)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_top_pairs(cm, ds, y, pred, prob, out, n_pairs=6, n_img=8):
    """가장 많이 헷갈린 (정답 → 예측) 쌍마다, 실제 틀린 사진을 확신도 높은 순으로 보여준다."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off.ravel())[::-1][:n_pairs]            # 오답 개수가 많은 칸 순서대로
    pairs = [divmod(k, len(CLASSES)) for k in flat]
    fig, axes = plt.subplots(n_pairs, n_img, figsize=(n_img * 1.4, n_pairs * 1.7))
    for r, (t, p) in enumerate(pairs):
        idx = np.where((y == t) & (pred == p))[0]
        idx = idx[np.argsort(-prob[idx, p])][:n_img]          # 틀린 답에 준 확률이 높은 순
        for c in range(n_img):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                i = idx[c]
                img = to_image(ds[i][0])
                ax.imshow(img if img.ndim == 2 else img.transpose(1, 2, 0), cmap=CMAP if img.ndim == 2 else None)
                ax.set_title(f"{prob[i, p] * 100:.0f}%", fontsize=8)
        axes[r, 0].text(-0.15, 0.5, f"{CLASSES[t]} → {CLASSES[p]}\n(n={cm[t, p]})", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle(f"{NAME}: most confused pairs (true → predicted). number above = probability given to the wrong class",
                 fontsize=10)
    fig.tight_layout(rect=(0.08, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confident(ds, y, pred, prob, out, n=24):
    """모델이 가장 확신하고 틀린 사진들"""
    wrong = np.where(y != pred)[0]
    wrong = wrong[np.argsort(-prob[wrong, pred[wrong]])][:n]  # 틀린 답에 준 확률이 높은 순
    cols = 8
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        img = to_image(ds[i][0])
        ax.imshow(img if img.ndim == 2 else img.transpose(1, 2, 0), cmap=CMAP if img.ndim == 2 else None)
        ax.set_title(f"T:{CLASSES[y[i]]}  P:{CLASSES[pred[i]]}\n{prob[i, pred[i]] * 100:.1f}%", fontsize=8)
    fig.suptitle(f"{NAME}: most confident mistakes (T = true, P = predicted, % = confidence)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="분석할 실행 폴더 (best.pt, history.json 이 있는 곳). 안 주면 자동 선택")
    args = p.parse_args()
    run = args.run or find_best_run()
    with open(os.path.join(run, "history.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    print(f"분석할 실행: {run}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=device))   # 저장된 가중치 불러오기
    model.to(device).eval()                                   # 평가 모드 (Dropout 끔)
    ds = make_test_set()
    y, pred, prob = predict(model, DataLoader(ds, batch_size=512), device)

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)    # 혼동 행렬
    np.add.at(cm, (y, pred), 1)                               # (정답, 예측) 칸마다 1씩 더하기

    os.makedirs("analysis", exist_ok=True)
    plot_confusion(cm, "analysis/1_confusion.png")
    plot_per_class(cm, "analysis/2_per_class.png")
    plot_top_pairs(cm, ds, y, pred, prob, "analysis/3_top_pairs.png")
    plot_confident(ds, y, pred, prob, "analysis/4_confident.png")

    # ---------------------------------------------- 터미널 요약
    wrong = (y != pred).sum()
    print(f"정확도 {(y == pred).mean() * 100:.2f}%  (틀린 사진 {wrong} / {len(y)}장)")
    acc = cm.diagonal() / cm.sum(1)
    print("클래스별 정확도 (낮은 순):", ", ".join(f"{CLASSES[i]} {acc[i] * 100:.1f}%" for i in np.argsort(acc)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("가장 많이 헷갈린 쌍:")
    for k in np.argsort(off.ravel())[::-1][:8]:
        t, q = divmod(k, len(CLASSES))
        print(f"  {CLASSES[t]:>10s} → {CLASSES[q]:<10s} {off[t, q]:4d}장  (오답 중 {off[t, q] / wrong * 100:.1f}%)")
    conf = prob[np.arange(len(y)), pred]                      # 예측한 클래스에 준 확률 = 확신도
    print(f"평균 확신도: 맞힌 사진 {conf[y == pred].mean() * 100:.1f}%, 틀린 사진 {conf[y != pred].mean() * 100:.1f}%")
    print("그림 저장: analysis/1_confusion.png ~ 4_confident.png")


if __name__ == "__main__":
    main()

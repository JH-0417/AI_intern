"""여러 실행의 history.json 을 읽어서, 하이퍼파라미터별 곡선을 한 그래프에 겹쳐 그린다.
같은 설정을 시드만 바꿔 여러 번 돌렸다면 평균선 + 흔들림 범위(음영, 평균 ± 표준편차)로 그린다.

사용 예:
    python plot.py runs/sweep_augment_1005_150000             # 그 폴더 안의 실행들을 비교
    python plot.py runs                                        # runs 아래 모든 실행을 비교
결과: 해당 폴더에 compare.png 저장 + 화면에 표시
"""
import argparse
import glob              # 패턴(*)으로 파일들을 한꺼번에 찾는 도구
import json
import os

import matplotlib.pyplot as plt   # 그래프 그리는 라이브러리
import numpy as np                # 평균, 표준편차 계산

KEYS = ["augment", "lr", "batch_size", "dropout"]   # 설정을 구분하는 하이퍼파라미터 (seed 는 제외 → 시드만 다른 실행은 같은 묶음)
CURVES = ["train_loss", "test_loss", "train_acc", "test_acc"]


def load_runs(folder):
    """folder 아래의 모든 history.json 을 읽어서 리스트로 돌려준다."""
    paths = sorted(glob.glob(os.path.join(folder, "**", "history.json"), recursive=True))   # 하위 폴더까지 전부 찾기
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))                         # JSON 파일 → 파이썬 사전
    return runs


def group_runs(runs):
    """시드만 다른 실행들을 하나의 묶음으로 모은다. 결과: {(augment, lr, batch, dropout): [실행, 실행, ...]}"""
    groups = {}
    for r in runs:
        key = tuple(r["config"][k] for k in KEYS)             # 시드를 뺀 설정값들
        groups.setdefault(key, []).append(r)                  # 같은 설정이면 같은 리스트에 추가
    return dict(sorted(groups.items()))                       # 설정값 순서대로 정렬 (예: lr 작은 것부터)


def plot_runs(folder):
    runs = load_runs(folder)
    if not runs:                                              # history.json 이 하나도 없으면
        print(f"{folder} 아래에 history.json 이 없습니다. 먼저 train.py 로 학습하세요.")
        return
    groups = group_runs(runs)
    keys = list(groups)                                       # 설정 목록
    # 설정마다 값이 다른 하이퍼파라미터만 범례에 표시 (모두 같은 값은 제목에 표시)
    diff = [i for i, k in enumerate(KEYS) if len({key[i] for key in keys}) > 1]
    common = [f"{k}={keys[0][i]}" for i, k in enumerate(KEYS) if i not in diff]
    seeds = sorted({r["config"]["seed"] for r in runs})       # 사용된 시드 목록

    def label(key):                                           # 범례 글자. 예: "augment=True"
        return ", ".join(f"{KEYS[i]}={key[i]}" for i in diff) or "base"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))           # 2x2 칸: 왼쪽 위부터 train loss, test loss, train acc, test acc
    titles = ["Train loss", "Test loss", "Train accuracy", "Test accuracy"]
    for ax, curve, title in zip(axes.flat, CURVES, titles):   # 칸 4개를 하나씩
        for n, (key, rs) in enumerate(groups.items(), 1):     # 설정마다 곡선 1개
            ys = np.array([r[curve] for r in rs])             # 모양: (시드 수, 에포크 수)
            mean, std = ys.mean(0), ys.std(0)                 # 에포크마다 시드들의 평균과 표준편차
            x = np.arange(1, ys.shape[1] + 1)                 # x축: 1, 2, ..., 에포크 수
            line, = ax.plot(x, mean, marker="o", markersize=3, label=f"[{n}] {label(key)}  (n={len(rs)})")
            if len(rs) > 1:                                   # 시드가 2개 이상일 때만 흔들림 범위를 음영으로
                ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)                                    # 옅은 격자
    axes[0, 0].legend()                                       # 범례는 첫 칸에만 (모든 칸이 같은 색 = 같은 설정)

    # 설정별 최고 test 정확도: 평균 ± 표준편차
    stats = []
    for key, rs in groups.items():
        best = np.array([r["best_test_acc"] for r in rs])
        stats.append((label(key), best.mean(), best.std(), len(rs)))
    summary = "   ".join(f"[{n}] {m:.4f}±{s:.4f}" for n, (_, m, s, _) in enumerate(stats, 1))
    fig.suptitle(f"CIFAR-10  ({', '.join(common)}, seeds={seeds})   line = mean, band = ±1 std\n"
                 f"best test acc  {summary}", fontsize=11)
    fig.tight_layout()                                        # 글자가 겹치지 않게 간격 자동 조절

    out = os.path.join(folder, "compare.png")
    fig.savefig(out, dpi=120)                                 # 그림 파일로 저장
    print(f"저장: {out}")
    for n, (lab, m, s, c) in enumerate(stats, 1):             # 터미널에도 요약 출력
        print(f"  [{n}] {lab:25s} best test acc {m:.4f} ± {s:.4f}  (시드 {c}개)")
    plt.show()                                                # 화면에 창으로 띄우기


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?", default="runs")       # 비교할 폴더 (안 쓰면 runs 전체)
    plot_runs(p.parse_args().folder)

"""하이퍼파라미터 하나를 여러 값으로 바꿔 가며 train.py 를 차례로 실행하고, 끝나면 비교 그래프를 그린다.
--seeds 로 시드를 여러 개 주면, 값마다 시드별로 반복 실행 → 그래프에 평균선 + 흔들림 범위로 표시된다.

사용 예:
    python sweep.py --param lr --values 0.01 0.001 0.0001
    python sweep.py --param dropout --values 0 0.25 0.5 --seeds 0 1 2      # 값 3개 x 시드 3개 = 9번 실행
    python sweep.py --param batch-size --values 32 128 512
"""
import argparse          # 터미널 옵션 받기
import subprocess        # 다른 파이썬 프로그램(train.py)을 실행하는 도구
import sys               # 지금 쓰고 있는 파이썬 실행 파일 경로(sys.executable)를 얻기 위해 사용
import time              # 실험 폴더 이름에 시각을 붙이기 위해 사용

from plot import plot_runs   # 같은 폴더의 plot.py 에서 그래프 그리는 함수 가져오기


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--param", required=True, choices=["lr", "batch-size", "dropout"])   # 바꿔 볼 하이퍼파라미터 이름
    p.add_argument("--values", required=True, nargs="+")      # 시험할 값들 (여러 개를 띄어쓰기로). 예: 0.01 0.001 0.0001
    p.add_argument("--seeds", type=int, nargs="+", default=[0])   # 값마다 반복할 시드들. 예: 0 1 2 (기본은 시드 0 하나)
    p.add_argument("--epochs", type=int, default=10)          # 실행마다 학습할 에포크 수
    p.add_argument("--workers", type=int, default=4)          # DataLoader 일꾼 수 (Windows 에서 오류 나면 0)
    args = p.parse_args()

    tag = f"sweep_{args.param}_{time.strftime('%m%d_%H%M%S')}"   # 이번 실험 묶음의 폴더 이름. 예: sweep_lr_1005_150000
    jobs = [(v, s) for v in args.values for s in args.seeds]  # 실행할 (값, 시드) 조합 전부. 예: 값 3개 x 시드 3개 = 9개
    for i, (v, s) in enumerate(jobs, 1):                      # 조합 하나씩 차례로 (i = 1, 2, 3 ...)
        cmd = [sys.executable, "train.py", "--epochs", str(args.epochs), "--workers", str(args.workers),
               "--seed", str(s), "--tag", tag, f"--{args.param}", v]
        # ↑ 실행할 명령어. 예: python train.py --epochs 10 --workers 4 --seed 1 --tag sweep_lr_... --lr 0.01
        print(f"\n===== [{i}/{len(jobs)}] --{args.param} {v}  seed {s} =====")
        subprocess.run(cmd, check=True)                       # train.py 실행하고 끝날 때까지 기다림. 오류가 나면 여기서 멈춤

    plot_runs(f"runs/{tag}")                                  # 모든 실행이 끝나면 한 그래프에 비교해서 그리기


if __name__ == "__main__":
    main()

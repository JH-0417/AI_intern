"""2주차 과제 1: CIFAR-10 사물 10종 CNN 분류 + TensorBoard 시각화

사용 예:
    python train.py --epochs 30 --augment
    python train.py --lr 0.01 --seed 1
    tensorboard --logdir runs
"""
# ============================================================ 라이브러리 불러오기
import argparse                     # 터미널에서 --epochs 20 처럼 옵션을 받을 수 있게 해주는 도구
import json                         # 에포크별 결과를 history.json 파일로 저장하기 위해 사용 (plot.py 가 읽음)
import random                       # Python 기본 랜덤 (시드 고정 대상)
import time                         # 현재 시각을 가져와 실행 폴더 이름에 붙이기 위해 사용

import numpy as np                  # numpy 랜덤 (시드 고정 대상)
import torch                        # PyTorch 본체
import torch.nn as nn               # 손실 함수(CrossEntropyLoss) 등
from torch.utils.data import DataLoader                 # Dataset 을 배치로 묶고 섞어서 꺼내주는 도구
from torch.utils.tensorboard import SummaryWriter       # TensorBoard 용 로그 파일을 기록하는 도구

from datasets import CIFAR10Dataset, Compose, RandomCropPad, RandomHFlip   # 같은 폴더의 datasets.py 에서 데이터셋과 증강 도구 가져오기
from model import SimpleCNN         # 같은 폴더의 model.py 에서 CNN 모델 클래스 가져오기


def set_seed(seed):
    """랜덤이 들어가는 모든 곳의 시드를 고정한다 → 같은 시드로 실행하면 같은 결과가 나온다 (재현성)."""
    # 랜덤이 쓰이는 곳: 가중치 초기값, DataLoader 의 shuffle 순서, Dropout 이 끄는 뉴런, 증강(크롭 위치, 반전 여부)
    random.seed(seed)                          # Python 기본 랜덤
    np.random.seed(seed)                       # numpy 랜덤
    torch.manual_seed(seed)                    # PyTorch CPU 랜덤 (가중치 초기화, shuffle, Dropout)
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU 랜덤
    torch.backends.cudnn.deterministic = True  # GPU 연산을 항상 같은 방식으로 하게 함 (조금 느려질 수 있음)
    torch.backends.cudnn.benchmark = False     # 실행마다 가장 빠른 방식을 새로 고르는 기능 끄기 (결과가 달라질 수 있어서)


def build(args):
    """데이터셋, 모델, DataLoader 를 만들어서 돌려준다."""
    aug = Compose([RandomCropPad(32, 4), RandomHFlip()]) if args.augment else None
    # ↑ --augment 를 주면 증강 사용: 4픽셀 패딩 후 랜덤 크롭(위치 이동) → 50% 확률 좌우 반전. 안 주면 None(증강 없음)
    train_set = CIFAR10Dataset(train=True, transform=aug)    # 학습용 5만 장 (증강은 학습셋에만)
    test_set = CIFAR10Dataset(train=False)                   # 테스트용 1만 장 (평가는 원본 그대로)
    model = SimpleCNN(in_channels=3, img_size=32, dropout=args.dropout)        # 컬러(RGB 3채널), 32x32 용 CNN 만들기
    g = torch.Generator().manual_seed(args.seed)
    # ↑ shuffle 순서 전용 랜덤 생성기. 시드를 고정해서 매번 같은 순서로 섞이게 함
    #   일꾼 프로세스들의 랜덤(증강)도 이 생성기에서 시드를 받아서, 증강 결과까지 재현됨
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                              generator=g)
    # ↑ 학습용 로더: 128장씩 묶음(batch_size), 매 에포크마다 순서 섞기(shuffle=True)
    #   num_workers: 데이터를 미리 준비하는 일꾼 프로세스 수, pin_memory: GPU 로 옮기는 속도 향상
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    # ↑ 테스트용 로더: 평가만 하니 섞을 필요 없음. 그래디언트를 저장 안 해서 메모리 여유가 있으니 배치를 2배(256)로
    return model, train_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizer 가 주어지면 학습, 아니면 평가 모드로 한 에포크 수행"""
    train = optimizer is not None         # optimizer 를 넘겼으면 True(학습), 안 넘겼으면 False(평가)
    model.train(train)                    # 학습 모드 / 평가 모드 전환 → Dropout 이 학습 때만 켜지고 평가 때는 꺼짐
    total_loss, correct, n = 0.0, 0, 0    # loss 합계, 맞힌 개수, 본 이미지 개수를 0 으로 시작
    with torch.set_grad_enabled(train):   # 학습 때만 그래디언트 계산을 켬. 평가 때는 꺼서 속도·메모리 절약
        for x, y in loader:               # 배치를 하나씩 꺼냄. x: 이미지 (128, 3, 32, 32), y: 정답 (128,)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)   # 데이터를 모델과 같은 장치(GPU/CPU)로 이동
            output = model(x)                 # 1. 예측: 이미지 128장 → 각각 사물 10종의 점수 (128, 10)
            loss = criterion(output, y)       # 2. 손실 계산: 예측이 정답과 얼마나 다른지 숫자 하나로
            if train:                         # 학습할 때만 아래 3단계 실행 (평가 때는 가중치를 바꾸면 안 됨)
                optimizer.zero_grad()         # 3. 그래디언트 초기화: 지난 배치의 그래디언트 지우기 (안 지우면 계속 더해짐)
                loss.backward()               # 4. 역전파: 모든 필터 값·가중치마다 "어느 방향으로 고칠지" 계산
                optimizer.step()              # 5. 가중치 업데이트: 그 방향으로 조금씩 실제로 수정
            total_loss += loss.item() * x.size(0)   # loss 는 배치 평균이라 배치 크기를 곱해서 합계로 누적 (.item(): 텐서 → 숫자)
            correct += (output.argmax(1) == y).sum().item()
            # ↑ argmax(1): 10개 점수 중 가장 큰 위치 = 예측한 클래스. 정답(y)과 같은 개수를 세서 누적
            n += x.size(0)                    # 이번 배치의 이미지 수 누적 (마지막 배치는 128보다 작을 수 있음)
    return total_loss / n, correct / n        # (평균 loss, 정확도) 반환. 예: (0.7345, 0.7491)


def main():
    # ---------------------------------------------- 터미널 옵션 받기
    p = argparse.ArgumentParser()                             # 옵션 해석기 만들기
    p.add_argument("--epochs", type=int, default=30)          # 학습 데이터 전체를 몇 번 반복할지 (기본 30, MNIST 보다 어려워서 더 길게)
    p.add_argument("--batch-size", type=int, default=128)     # 한 번에 넣을 이미지 수 (기본 128)
    p.add_argument("--lr", type=float, default=1e-3)          # 학습률: 한 번에 가중치를 얼마나 크게 고칠지 (기본 0.001)
    p.add_argument("--dropout", type=float, default=0.5)      # Dropout 비율 (0 이면 Dropout 없음)
    p.add_argument("--workers", type=int, default=4)          # DataLoader 일꾼 프로세스 수 (기본 4)
    p.add_argument("--augment", action="store_true", help="CIFAR-10 학습 시 RandomCrop + HFlip 적용")   # 이 옵션을 쓰면 증강 켜짐
    p.add_argument("--seed", type=int, default=0)             # 랜덤 시드. 같은 시드 + 같은 설정 = 같은 결과
    p.add_argument("--tag", type=str, default="")             # 실행 폴더 이름 앞에 붙일 이름 (sweep.py 가 실험별로 묶을 때 사용)
    args = p.parse_args()                                     # 실제 입력값 읽기 → args.epochs, args.batch_size ... 로 사용
    set_seed(args.seed)                                       # 모델·데이터를 만들기 "전에" 시드 고정 (가중치 초기값부터 같아지도록)

    # ---------------------------------------------- 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU 가 있으면 GPU, 없으면 CPU 사용
    print(f"device: {device}")                                # 어떤 장치로 학습하는지 출력
    model, train_loader, test_loader = build(args)            # 데이터와 모델 만들기
    model = model.to(device)                                  # 모델의 모든 가중치를 GPU(또는 CPU)로 이동
    criterion = nn.CrossEntropyLoss()                         # 분류용 손실 함수 (내부에 softmax 포함)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ↑ Adam: 그래디언트를 보고 가중치를 고치는 방법. model.parameters() = 고칠 대상(필터 값, Linear 가중치 전부)

    hp = f"{'aug' if args.augment else 'noaug'}_lr{args.lr}_bs{args.batch_size}_do{args.dropout}_seed{args.seed}"
    # ↑ 하이퍼파라미터를 이름으로. 예: aug_lr0.001_bs128_do0.5_seed0
    run_name = f"runs/{args.tag + '/' if args.tag else ''}cifar10_{hp}_{time.strftime('%m%d_%H%M%S')}"
    # ↑ 이번 실행의 저장 폴더. 예: runs/cifar10_aug_lr0.001_bs128_do0.5_seed0_1005_143012
    #   폴더 이름에 설정이 들어 있어서 TensorBoard 에서 어떤 곡선이 어떤 설정인지 바로 알 수 있음
    writer = SummaryWriter(run_name)                          # TensorBoard 기록기 만들기 (폴더가 자동 생성됨)
    history = {"config": vars(args), "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    # ↑ 에포크별 결과를 모아둘 사전. 끝나면 history.json 으로 저장 → plot.py 가 이걸 읽어서 비교 그래프를 그림

    # ---------------------------------------------- 학습 반복
    best_acc = 0.0                                            # 지금까지 가장 높은 테스트 정확도
    for epoch in range(1, args.epochs + 1):                   # 1 ~ 30 에포크 반복
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)   # 학습 (optimizer 넘김 → 학습 모드)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device)               # 평가 (optimizer 안 넘김 → 평가 모드)

        writer.add_scalar("Loss/train", tr_loss, epoch)       # TensorBoard 에 기록 (태그 이름, 값, x축=epoch)
        writer.add_scalar("Loss/test", te_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/test", te_acc, epoch)
        # ↑ add_scalars 대신 add_scalar 를 쓰는 이유: 실행 여러 개를 TensorBoard 에서 겹쳐 볼 때
        #   실행 1개 = 곡선 1개(이름 = 폴더 이름)로 깔끔하게 나옴. add_scalars 는 하위 폴더가 생겨 범례가 지저분해짐
        for k, v in (("train_loss", tr_loss), ("test_loss", te_loss), ("train_acc", tr_acc), ("test_acc", te_acc)):
            history[k].append(v)                              # 이번 에포크 결과를 history 에 추가
        print(f"[{epoch:3d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}", flush=True)
        # ↑ 터미널 출력. :3d = 3자리 정수, :.4f = 소수점 4자리. flush=True: 로그 파일에도 바로바로 기록

        if te_acc > best_acc:                                 # 테스트 정확도가 최고 기록을 넘으면
            best_acc = te_acc                                 # 최고 기록 갱신
            torch.save(model.state_dict(), f"{run_name}/best.pt")
            # ↑ 현재 가중치(필터 값 등)를 best.pt 파일로 저장. 과적합으로 나중에 나빠져도 최고 시점 모델이 남음

    writer.close()                                            # 기록기 닫기 (남은 로그를 파일에 확실히 저장)
    history["best_test_acc"] = best_acc                       # 최고 정확도도 함께 저장
    with open(f"{run_name}/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)                       # history 를 사람이 읽을 수 있는 JSON 파일로 저장
    print(f"best test acc: {best_acc:.4f}  (logs: {run_name})")   # 최종 결과와 로그 폴더 위치 출력


if __name__ == "__main__":   # 이 파일을 직접 실행했을 때만 main() 실행 (다른 파일에서 import 할 때는 실행 안 됨)
    main()

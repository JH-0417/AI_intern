"""2주차: AlexNet 논문 구조 구현 + Imagenette(ImageNet 10클래스 부분집합) 학습

사용 예 (week2/alexnet 폴더에서):
    python train.py                      # 기본: ReLU + LRN + Dropout + GPU 분할(groups=2)
    python train.py --act tanh           # 논문 Figure 1: ReLU vs tanh 학습 속도 비교
    python train.py --dropout 0          # Dropout 제거 → 과적합 비교
    python train.py --no-lrn             # LRN 효과 비교
    python train.py --init paper         # 논문 초기화 N(0, 0.01) + bias 1
    tensorboard --logdir runs
"""
import argparse
import os
import tarfile
import time
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import AlexNet

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"   # 약 340MB
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # 논문은 위치별 평균 이미지를 뺐지만, 현재 표준인 채널별 정규화 사용


def prepare_imagenette(root="data"):
    """Imagenette 를 내려받아 압축을 풀고 train/val 폴더 경로를 돌려준다. 이미 있으면 건너뜀"""
    folder = os.path.join(root, "imagenette2-320")
    if not os.path.isdir(folder):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "imagenette2-320.tgz")
        if not os.path.exists(tgz):
            print(f"Downloading {IMAGENETTE_URL}")
            urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        print("Extracting...")
        with tarfile.open(tgz) as f:
            f.extractall(root)
    return os.path.join(folder, "train"), os.path.join(folder, "val")


def build(args):
    train_dir, val_dir = prepare_imagenette()
    # 논문 4.1: 짧은 변 256 리사이즈 → 랜덤 크롭 + 좌우 반전 (227 은 구조 계산이 맞는 크기)
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = datasets.ImageFolder(train_dir, train_tf)   # 폴더 이름 = 클래스
    val_set = datasets.ImageFolder(val_dir, val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = AlexNet(num_classes=len(train_set.classes), act=args.act, lrn=not args.no_lrn,
                    dropout=args.dropout, split_gpu=not args.no_split)
    if args.init == "paper":
        model.init_paper()
    return model, train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizer 가 주어지면 학습, 아니면 평가 모드로 한 에포크 수행"""
    train = optimizer is not None
    model.train(train)                        # Dropout 은 train 일 때만 동작
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(x)                 # 1. 예측
            loss = criterion(output, y)       # 2. 손실 계산
            if train:
                optimizer.zero_grad()         # 3. 그래디언트 초기화
                loss.backward()               # 4. 역전파
                optimizer.step()              # 5. 가중치 업데이트
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)               # 논문 128
    p.add_argument("--lr", type=float, default=0.01)                    # 논문 0.01
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--act", choices=["relu", "tanh"], default="relu")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no-lrn", action="store_true")
    p.add_argument("--no-split", action="store_true", help="groups=2(GPU 분할) 없이 전체 연결")
    p.add_argument("--init", choices=["default", "paper"], default="default",
                   help="default: PyTorch 기본(Kaiming), paper: N(0, 0.01) + bias 1")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, train_loader, val_loader = build(args)
    model = model.to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    # 논문 5장: SGD, momentum 0.9, weight decay 0.0005
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 논문: 검증 에러가 더 안 줄면 학습률 / 10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    tag = f"{args.act}{'_nolrn' if args.no_lrn else ''}_do{args.dropout}{'_nosplit' if args.no_split else ''}_{args.init}"
    run_name = f"runs/{tag}_{time.strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(run_name)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        lr = optimizer.param_groups[0]["lr"]

        writer.add_scalars("Loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc, "val": va_acc}, epoch)
        writer.add_scalar("LR", lr, epoch)
        print(f"[{epoch:3d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.0e} | {time.time() - t0:.0f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), f"{run_name}/best.pt")

    writer.close()
    print(f"best val acc: {best_acc:.4f}  (logs: {run_name})")


if __name__ == "__main__":
    main()

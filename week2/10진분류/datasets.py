"""CIFAR-10 원본 바이너리 파일을 직접 파싱하는 Custom Dataset.

torchvision.datasets 를 쓰지 않고, 공식 배포 파일을 내려받아 numpy 로 바이트 단위 파싱한다.
"""
# ============================================================ 라이브러리 불러오기
import os                   # 폴더 만들기, 파일 경로 합치기, 파일이 있는지 확인하는 도구
import tarfile              # .tar.gz 묶음 파일을 푸는 도구 (CIFAR-10 파일이 이 형식)
import urllib.request       # 인터넷에서 파일을 내려받는 도구

import numpy as np          # 바이트 덩어리를 숫자 배열로 바꾸고 모양(reshape)을 바꾸는 데 사용
import torch                # PyTorch: 배열을 텐서로 바꾸고 모델에 넣기 위해 사용
from torch.utils.data import Dataset   # 우리가 만들 데이터셋 클래스의 부모 클래스

# ============================================================ 다운로드 주소와 파일 이름
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"       # CIFAR-10 바이너리 버전 다운로드 주소
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",          # 라벨 숫자 0~9 가 뜻하는 이름
                 "dog", "frog", "horse", "ship", "truck"]                   # 예: 라벨 3 = cat (고양이)


def download(url, path):
    """url 의 파일을 path 에 저장한다. 이미 있으면 다시 받지 않는다."""
    if not os.path.exists(path):                              # 저장할 위치에 파일이 아직 없을 때만
        os.makedirs(os.path.dirname(path), exist_ok=True)     # 파일이 들어갈 폴더를 만든다 (이미 있으면 그냥 넘어감)
        print(f"Downloading {url}")                           # 다운로드 시작을 화면에 알림
        urllib.request.urlretrieve(url, path)                 # 인터넷에서 내려받아 path 에 저장
    return path                                               # 파일 경로를 돌려줌 → 바로 다음 함수에 넘겨 쓰기 편함


# ---------------------------------------------------------------- CIFAR-10
def read_cifar_bin(path):
    """각 레코드 3073B: [label(1B)][R 1024B][G 1024B][B 1024B] (32x32, row-major)"""
    # 헤더 없음. 이미지 1장 = 라벨 1바이트 + 빨강 32x32 + 초록 32x32 + 파랑 32x32 = 3073바이트가 계속 반복됨
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 3073)
    # ↑ 파일 전체를 1바이트씩 읽고, 3073바이트씩 한 줄로 자른다 → 모양 (10000, 3073) = 한 줄이 이미지 1장
    #   -1 = "개수는 알아서 계산해" (전체 크기 ÷ 3073)
    return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0]
    # ↑ raw[:, 1:] = 각 줄의 1번~끝(3072바이트) = 이미지 → (10000, 3, 32, 32) 로 모양 변경 (R, G, B 순서라 채널이 맨 앞)
    #   raw[:, 0]  = 각 줄의 0번(첫 바이트) = 라벨 → (10000,)


class CIFAR10Dataset(Dataset):
    """CIFAR-10 컬러 사진 데이터셋 (사물 10종, 32x32, RGB 3채널)."""
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)   # R, G, B 채널별 평균
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)    # R, G, B 채널별 표준편차
    # ↑ .view(3, 1, 1): (3,) → (3, 1, 1) 모양으로. 그래야 (3, 32, 32) 이미지에서 빼고 나눌 때
    #   채널마다 자기 값이 32x32 전체에 적용됨 (브로드캐스팅)

    def __init__(self, root="data/cifar10", train=True, transform=None):
        # [3대 요소 ①] 파일을 읽어서 메모리에 올려두기
        base = os.path.join(root, "cifar-10-batches-bin")   # 압축을 풀면 생기는 폴더 경로: data/cifar10/cifar-10-batches-bin
        if not os.path.isdir(base):                          # 그 폴더가 없으면 (= 아직 안 받았거나 안 풀었으면)
            with tarfile.open(download(CIFAR_URL, os.path.join(root, "cifar-10-binary.tar.gz"))) as tar:   # 내려받고 열기
                tar.extractall(root)                         # data/cifar10 안에 압축 풀기
        files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        # ↑ 학습용이면 data_batch_1.bin ~ data_batch_5.bin (각 1만 장), 테스트용이면 test_batch.bin (1만 장)
        parts = [read_cifar_bin(os.path.join(base, f)) for f in files]   # 파일마다 파싱 → [(이미지, 라벨), (이미지, 라벨), ...]
        self.images = np.concatenate([p[0] for p in parts])   # 이미지들을 하나로 이어 붙이기 → (50000, 3, 32, 32)
        self.labels = np.concatenate([p[1] for p in parts])   # 라벨들도 이어 붙이기 → (50000,)
        self.transform = transform                            # 증강 함수 저장 (--augment 를 주면 들어옴, 아니면 None)

    def __len__(self):
        # [3대 요소 ②] 전체 개수
        return len(self.labels)                       # 학습셋 50000, 테스트셋 10000

    def __getitem__(self, idx):
        # [3대 요소 ③] idx 번째 (이미지, 라벨) 꺼내기
        x = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)
        # ↑ idx 번째 이미지 (3, 32, 32) → 소수로 → 0~1 범위 → 텐서. 이미 채널 축이 있어서 unsqueeze 필요 없음
        if self.transform:  # 증강은 정규화 전 [0,1] 범위에서 적용
            x = self.transform(x)
            # ↑ 이유: 증강(패딩)은 빈칸을 0으로 채우는데, 정규화 전 0 = 검은색이지만 정규화 후 0 = 평균색이 되어 의미가 바뀜
        x = (x - self.MEAN) / self.STD                # 채널별 정규화 (R 은 R 평균으로, G 는 G 평균으로 ...)
        return x, int(self.labels[idx])               # (이미지 텐서, 정답 번호) 반환. 예: (텐서, 3) → cat


# ---------------------------------------------------------------- 간단한 증강 (1주차 과제 2 방식, 텐서만 사용)
# 증강 = 학습할 때마다 이미지를 조금씩 바꿔서 넣는 것 → 같은 사진을 외우지 못하게 해서 과적합을 줄임
class RandomHFlip:
    """확률 p 로 이미지를 좌우 반전한다."""
    def __init__(self, p=0.5):
        self.p = p                                    # 반전할 확률 (0.5 = 절반은 반전, 절반은 그대로)

    def __call__(self, x):
        # __call__ 이 있으면 객체를 함수처럼 쓸 수 있음: flip = RandomHFlip();  x = flip(x)
        return torch.flip(x, dims=[2]) if torch.rand(1).item() < self.p else x
        # ↑ torch.rand(1): 0~1 사이 랜덤 숫자 1개. 그게 p 보다 작으면 반전, 아니면 원본 그대로
        #   dims=[2]: x 모양이 (채널=0, 세로=1, 가로=2) 이므로 2번 축(가로)을 뒤집음 = 좌우 반전


class RandomCropPad:
    """padding 후 원래 크기로 랜덤 크롭"""
    # 효과: 사물이 상하좌우로 최대 padding 픽셀만큼 무작위로 이동한 것처럼 보이게 함
    def __init__(self, size=32, padding=4):
        self.size, self.padding = size, padding       # 잘라낼 크기 32, 테두리에 덧붙일 두께 4

    def __call__(self, x):
        x = torch.nn.functional.pad(x, [self.padding] * 4)
        # ↑ [4, 4, 4, 4] = (왼쪽, 오른쪽, 위, 아래)에 0을 4픽셀씩 덧붙임 → (3, 32, 32) 가 (3, 40, 40) 이 됨
        i, j = torch.randint(0, 2 * self.padding + 1, (2,)).tolist()
        # ↑ 0~8 사이 랜덤 정수 2개 뽑기: i = 위에서 몇 칸 내려서 자를지, j = 왼쪽에서 몇 칸 옮겨서 자를지
        return x[:, i:i + self.size, j:j + self.size]
        # ↑ 40x40 에서 (i, j) 위치부터 32x32 만큼 잘라냄. 채널(:)은 전부 그대로


class Compose:
    """여러 증강을 순서대로 이어서 적용한다 (torchvision.transforms.Compose 를 직접 구현)."""
    def __init__(self, transforms):
        self.transforms = transforms                  # 증강 함수 리스트 저장. 예: [RandomCropPad(), RandomHFlip()]

    def __call__(self, x):
        for t in self.transforms:                     # 리스트 앞에서부터 하나씩
            x = t(x)                                  # 적용하고, 결과를 다음 증강에 넘김
        return x                                      # 모든 증강을 거친 이미지 반환

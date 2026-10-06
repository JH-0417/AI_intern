"""MNIST 원본 바이너리(IDX) 파일을 직접 파싱하는 Custom Dataset.

torchvision.datasets 를 쓰지 않고, 공식 배포 파일을 내려받아 numpy 로 바이트 단위 파싱한다.
"""
# ============================================================ 라이브러리 불러오기
import gzip                 # .gz 압축 파일을 풀어서 읽는 도구 (MNIST 파일이 .gz 로 압축돼 있음)
import os                   # 폴더 만들기, 파일 경로 합치기, 파일이 있는지 확인하는 도구
import urllib.request       # 인터넷에서 파일을 내려받는 도구

import numpy as np          # 바이트 덩어리를 숫자 배열로 바꾸고 모양(reshape)을 바꾸는 데 사용
import torch                # PyTorch: 배열을 텐서로 바꾸고 모델에 넣기 위해 사용
from torch.utils.data import Dataset   # 우리가 만들 데이터셋 클래스의 부모 클래스

# ============================================================ 다운로드 주소와 파일 이름
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"   # MNIST 파일들이 있는 인터넷 주소(폴더)
MNIST_FILES = {                                                     # 학습용/테스트용 파일 이름을 묶어둔 사전(dict)
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),   # 학습용: (이미지 파일, 라벨 파일)
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),      # 테스트용: (이미지 파일, 라벨 파일), t10k = 1만 장
}


def download(url, path):
    """url 의 파일을 path 에 저장한다. 이미 있으면 다시 받지 않는다."""
    if not os.path.exists(path):                              # 저장할 위치에 파일이 아직 없을 때만
        os.makedirs(os.path.dirname(path), exist_ok=True)     # 파일이 들어갈 폴더를 만든다 (이미 있으면 그냥 넘어감)
        print(f"Downloading {url}")                           # 다운로드 시작을 화면에 알림
        urllib.request.urlretrieve(url, path)                 # 인터넷에서 내려받아 path 에 저장
    return path                                               # 파일 경로를 돌려줌 → 바로 다음 함수에 넘겨 쓰기 편함


# ---------------------------------------------------------------- MNIST
def read_idx_images(path):
    """IDX3 형식: [magic(4B)=2051][N(4B)][rows(4B)][cols(4B)][pixels...] (big-endian)"""
    # 파일 구조: 앞 16바이트(헤더) = 숫자 4개(각 4바이트), 그 뒤 = 픽셀값(1픽셀 = 1바이트, 0~255)
    with gzip.open(path, "rb") as f:          # .gz 압축을 풀면서 파일 열기 ("rb" = 바이너리 읽기 모드)
        data = f.read()                       # 파일 전체를 바이트 덩어리로 한 번에 읽기
    magic, n, rows, cols = np.frombuffer(data, dtype=">i4", count=4)
    # ↑ 앞에서부터 4바이트 정수 4개를 읽는다.
    #   ">i4" = ">"(big-endian: 큰 자리부터 저장) + "i"(정수) + "4"(4바이트)
    #   결과: magic=2051, n=60000(이미지 개수), rows=28(세로), cols=28(가로)
    assert magic == 2051, f"잘못된 magic number: {magic}"   # 2051 이 아니면 이미지 파일이 아님 → 바로 오류 내고 멈춤
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    # ↑ 헤더 16바이트를 건너뛰고(offset=16) 나머지를 1바이트씩(uint8, 0~255) 읽은 뒤
    #   (60000, 28, 28) 모양으로 바꿔서 돌려줌 = 28x28 이미지 6만 장


def read_idx_labels(path):
    """IDX1 형식: [magic(4B)=2049][N(4B)][labels...]"""
    # 파일 구조: 앞 8바이트(헤더) = 숫자 2개, 그 뒤 = 라벨(1개 = 1바이트, 0~9)
    with gzip.open(path, "rb") as f:          # 압축 풀면서 열기
        data = f.read()                       # 전체를 바이트로 읽기
    magic, n = np.frombuffer(data, dtype=">i4", count=2)    # 앞 8바이트 → magic=2049, n=60000(라벨 개수)
    assert magic == 2049, f"잘못된 magic number: {magic}"   # 2049 가 아니면 라벨 파일이 아님 → 오류
    return np.frombuffer(data, dtype=np.uint8, offset=8)    # 헤더 8바이트 뒤부터 1바이트씩 = 라벨 6만 개 (모양: (60000,))


class MNISTDataset(Dataset):
    """MNIST 손글씨 숫자 데이터셋. DataLoader 에 넣으면 (이미지, 라벨)을 배치로 꺼내준다."""
    MEAN, STD = 0.1307, 0.3081    # MNIST 학습셋 전체 픽셀의 평균과 표준편차 (정규화에 사용)

    def __init__(self, root="data/mnist", train=True, transform=None):
        # [3대 요소 ①] 처음 만들 때 1번 실행: 파일을 읽어서 메모리에 올려둔다
        img_file, lbl_file = MNIST_FILES["train" if train else "test"]   # train=True 면 학습용, False 면 테스트용 파일 이름 고르기
        self.images = read_idx_images(download(MNIST_URL + img_file, os.path.join(root, img_file)))
        # ↑ (1) 이미지 파일 다운로드(없을 때만) → (2) 파싱 → (3) self.images 에 저장. 모양: (60000, 28, 28)
        self.labels = read_idx_labels(download(MNIST_URL + lbl_file, os.path.join(root, lbl_file)))
        # ↑ 라벨도 같은 방식. 모양: (60000,)
        assert len(self.images) == len(self.labels)   # 이미지 수와 라벨 수가 같은지 확인 (다르면 짝이 안 맞음)
        self.transform = transform                    # 증강 함수 저장 (MNIST 에서는 안 써서 None)

    def __len__(self):
        # [3대 요소 ②] 데이터가 총 몇 개인지 알려준다 → DataLoader 가 배치를 몇 번 만들지 계산할 때 사용
        return len(self.labels)                       # 학습셋 60000, 테스트셋 10000

    def __getitem__(self, idx):
        # [3대 요소 ③] idx 번째 (이미지, 라벨) 1쌍을 꺼내준다. DataLoader 가 이걸 128번 불러서 배치를 만든다
        # (28, 28) uint8 -> (1, 28, 28) float, 정규화
        x = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0).unsqueeze(0)
        # ↑ (1) idx 번째 이미지 꺼내기 (28, 28), 값 0~255
        #   (2) astype(float32): 정수 → 소수로 (나눗셈과 학습을 위해)
        #   (3) / 255.0: 0~255 → 0~1 범위로
        #   (4) torch.from_numpy: numpy 배열 → PyTorch 텐서
        #   (5) unsqueeze(0): 맨 앞에 채널 축 추가 (28, 28) → (1, 28, 28). Conv2d 는 (채널, 세로, 가로)를 원함
        x = (x - self.MEAN) / self.STD                # 정규화: 평균을 빼고 표준편차로 나눔 → 값이 0 근처에 모여 학습이 안정적
        if self.transform:                            # 증강 함수가 있으면
            x = self.transform(x)                     # 적용 (MNIST 는 None 이라 실행 안 됨)
        return x, int(self.labels[idx])               # (이미지 텐서, 정답 숫자) 반환. 예: (텐서, 5)

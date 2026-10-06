import torch.nn as nn   # nn = neural network. Conv2d, Linear 같은 레이어들이 들어 있는 모듈


class SimpleCNN(nn.Module):   # nn.Module 을 상속 → parameters(), to(device), train()/eval(), state_dict() 를 자동으로 쓸 수 있음
    """[Conv2d -> ReLU -> MaxPool2d] x 3  ->  Flatten -> Linear -> ReLU -> Linear"""

    def __init__(self, in_channels=1, num_classes=10, img_size=28, channels=(32, 64, 128), hidden=256, dropout=0.5):
        # 모델을 만들 때 1번 실행: 사용할 레이어(부품)를 준비하는 곳
        #   in_channels : 입력 채널 수 (MNIST 흑백 = 1)
        #   num_classes : 맞힐 정답 종류 수 (숫자 0~9 = 10)
        #   img_size    : 입력 이미지 한 변의 길이 (MNIST = 28)
        #   channels    : Conv 블록마다 필터 개수 (32개 → 64개 → 128개). 개수 = 블록 반복 횟수(3회)
        #   hidden      : 첫 번째 Linear 의 출력 뉴런 수
        #   dropout     : Dropout 으로 끌 뉴런 비율 (0.5 = 절반)
        super().__init__()        # 부모 클래스(nn.Module) 초기화. 이게 없으면 레이어 등록이 안 돼서 오류

        # ---------------------------------------------- 특징 추출부: [Conv → ReLU → MaxPool] 블록 3개
        layers, c = [], in_channels   # layers = 레이어를 담을 빈 리스트, c = 현재 입력 채널 수 (처음엔 1)
        for out_c in channels:        # out_c 가 32 → 64 → 128 순서로 바뀌며 3번 반복
            layers += [nn.Conv2d(c, out_c, kernel_size=3, padding=1),   # 3x3 필터 out_c 개로 패턴 찾기. padding=1 이라 크기 유지
                       nn.ReLU(inplace=True),                           # 음수 → 0 (비선형성 추가). inplace=True: 메모리 절약
                       nn.MaxPool2d(2)]                                 # 2x2 구역에서 최댓값만 → 가로세로 절반
            c = out_c                 # 이번 블록의 출력 채널 = 다음 블록의 입력 채널 (1→32, 32→64, 64→128)
        self.features = nn.Sequential(*layers)
        # ↑ 리스트에 담긴 레이어 9개를 순서대로 이어서 하나의 모듈로 묶음. *layers = 리스트를 풀어서 하나씩 넘김
        #   크기 변화: (1, 28, 28) → (32, 14, 14) → (64, 7, 7) → (128, 3, 3)

        # ---------------------------------------------- 분류부: Flatten → Linear → ReLU → Dropout → Linear
        feat_size = img_size // (2 ** len(channels))  # MNIST 28->14->7->3
        # ↑ MaxPool 을 3번(len(channels)) 거치면 크기가 2³=8 로 나뉨. // 는 나머지를 버리는 나눗셈 → 28 // 8 = 3
        self.classifier = nn.Sequential(
            nn.Flatten(),                                       # (128, 3, 3) → 한 줄로 펴기 → 1152 개
            nn.Linear(c * feat_size * feat_size, hidden),       # 1152(=128x3x3) 개 → 256 개. 특징들을 조합
            nn.ReLU(inplace=True),                              # 비선형성 (없으면 Linear 2개 = Linear 1개와 같아짐)
            nn.Dropout(dropout),                                # 학습 때만 뉴런 절반을 무작위로 끔 → 과적합 방지
            nn.Linear(hidden, num_classes),                     # 256 개 → 10 개 = 숫자 0~9 각각의 점수(logit)
        )
        # ↑ 마지막에 softmax 를 넣지 않는 이유: CrossEntropyLoss 가 안에서 softmax 를 같이 계산하기 때문

    def forward(self, x):
        # 데이터가 흐르는 순서를 정의. model(x) 라고 부르면 자동으로 이 함수가 실행됨
        # x: (배치, 1, 28, 28) → features → (배치, 128, 3, 3) → classifier → (배치, 10)
        return self.classifier(self.features(x))   # 특징 추출 후 분류. backward(역전파)는 PyTorch 가 자동으로 만들어 줌

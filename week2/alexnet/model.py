"""AlexNet (Krizhevsky et al., 2012) 논문 구조 그대로 구현

- GPU 2장 분할은 groups=2 로 재현 (Conv2, Conv4, Conv5 는 같은 GPU 채널만 봄)
- Conv1, Conv2 뒤 LRN, Conv1/2/5 뒤 겹치는 MaxPool(3, stride 2)
- 입력 227x227 (논문 그림의 224 는 계산이 맞지 않음)
"""
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, act="relu", lrn=True, dropout=0.5, split_gpu=True):
        super().__init__()
        Act = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]
        g = 2 if split_gpu else 1
        LRN = (lambda: nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)) if lrn else nn.Identity

        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96 -> pool 27x27x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4), Act(), LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 27x27x96 -> 27x27x256 -> pool 13x13x256 (GPU 별 분리)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=g), Act(), LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3: 13x13x256 -> 13x13x384 (GPU 교차)
            nn.Conv2d(256, 384, kernel_size=3, padding=1), Act(),
            # Conv4: 13x13x384 -> 13x13x384 (GPU 별 분리)
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=g), Act(),
            # Conv5: 13x13x384 -> 13x13x256 -> pool 6x6x256 (GPU 별 분리)
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=g), Act(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # 6x6x256 = 9216
            nn.Linear(256 * 6 * 6, 4096), Act(), nn.Dropout(dropout),   # FC6
            nn.Linear(4096, 4096), Act(), nn.Dropout(dropout),          # FC7
            nn.Linear(4096, num_classes),                               # FC8 (Softmax 는 CrossEntropyLoss 안에서)
        )

    def init_paper(self):
        """논문 초기화: 가중치 N(0, 0.01), bias 는 Conv2/4/5 와 FC6/7(은닉층) 1, 나머지 0"""
        convs = [m for m in self.features if isinstance(m, nn.Conv2d)]
        fcs = [m for m in self.classifier if isinstance(m, nn.Linear)]
        for i, m in enumerate(convs + fcs):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0 if i in (0, 2, 7) else 1.0)   # Conv1, Conv3, FC8 은 0
        return self

    def forward(self, x):
        return self.classifier(self.features(x))

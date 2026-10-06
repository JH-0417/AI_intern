# 딥러닝 인턴십 (4주)

PyTorch로 딥러닝의 핵심 원리를 직접 구현하며 익히는 4주 과정의 코드와 결과물.

| 주차 | 내용 | 폴더 |
|---|---|---|
| Week 1 | PyTorch 기초, 이미지 증강 직접 구현, 고양이·강아지 Custom Dataset | `week1/` |
| Week 2 | MNIST / CIFAR-10 CNN 분류기, 하이퍼파라미터 실험, 오답 분석, AlexNet | `week2/` |

## 환경

```bash
conda create -n intern python=3.10
conda activate intern
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tensorboard
```

## Week 2 구성

```
week2/
├── 손글씨/        MNIST (IDX 파일 직접 파싱 → SimpleCNN)
├── 10진분류/      CIFAR-10 (바이너리 직접 파싱 + 증강)
├── alexnet/       AlexNet 논문 구조 구현 (Imagenette)
└── Week2_랩미팅.pptx
```

각 폴더 안에서 실행한다 (데이터는 처음 실행할 때 `data/`에 자동으로 내려받음).

```bash
python train.py                                               # 기본 학습
python sweep.py --param dropout --values 0 0.25 0.5 --seeds 0 1 2   # 하이퍼파라미터 비교 (시드 3개)
python plot.py runs/sweep_...                                 # 비교 그래프
python analyze.py                                             # 오답 분석 (혼동 행렬, 틀린 사진)
tensorboard --logdir runs
```

데이터셋(`data/`)과 체크포인트(`*.pt`)는 용량 때문에 저장소에서 제외했다. 실험 기록(`history.json`, TensorBoard 로그)과 그래프(`compare.png`, `analysis/`)는 포함한다.

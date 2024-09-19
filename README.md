# Pause Token을 활용한 한국어 언어모델 Fine-tuning

이 프로젝트는 논문 [**"THINK BEFORE YOU SPEAK: TRAINING LANGUAGE MODELS WITH PAUSE TOKENS"**](https://arxiv.org/abs/2310.02226)을 기반으로, 한국어 언어모델에 Pause Token을 적용하여 Fine-tuning하는 코드입니다.

## 프로젝트 개요

언어모델의 학습 과정에서 Pause Token을 활용하여 모델의 성능을 향상시키는 방법을 연구합니다. 영어로 작성된 원 논문의 아이디어를 한국어에 맞게 적용하였으며, 한국어 접두사를 활용하여 Pause Token을 삽입하는 방식으로 전처리를 수행합니다.

## 사용한 모델과 데이터셋

### 사용한 모델: SKT KoGPT2

- **모델명**: `skt/kogpt2-base-v2`
- **설명**: KoGPT2는 SK텔레콤에서 공개한 한국어 GPT-2 언어모델로, 한국어 데이터로 사전 학습된 모델입니다. 이 모델을 기반으로 Pause Token을 적용하여 추가 학습을 진행합니다.
- **특징**:
  - 한국어 위키백과, 뉴스, 책 등 대용량 코퍼스로 사전 학습
  - 다양한 한국어 자연어 처리 태스크에 적용 가능
- **모델 링크**: [skt/kogpt2-base-v2 (Hugging Face)](https://huggingface.co/skt/kogpt2-base-v2)

### 사용한 데이터셋

#### 1. KorQuAD (Korean Question Answering Dataset)

- **구성**:
  - 학습 데이터: 약 6만 개의 질의응답 쌍
  - 검증 데이터: 약 5천 개의 질의응답 쌍

#### 2. KLUE NLI (Korean Language Understanding Evaluation - Natural Language Inference)

- **구성**:
  - 학습 데이터: 약 25,000개의 문장 쌍
  - 검증 데이터: 약 3,000개의 문장 쌍

## 주요 기능

- **모델 및 토크나이저 로딩**: `model.py`
- **데이터 전처리**: `preprocessing.py`
- **Custom Trainer 구현**: `trainer.py`
- **모델 평가**: `evaluation.py`
- **유틸리티 함수**: `utils.py`
- **메인 실행 스크립트**: `main.py`

## 실행 방법

### 1. 환경 설정

#### 아나콘다 가상환경 생성 및 활성화

```bash
# 가상환경 생성 (예: pause_env)
conda create -n pause_env python=3.8

# 가상환경 활성화
conda activate pause_env
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

주의: torch는 CUDA 버전 등에 따라 설치 방법이 다르므로, 직접 설치하시기 바랍니다.
예를 들어, CUDA 11.1을 사용하는 경우:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```


### 3. 코드 실행
main.py를 실행하여 모델을 학습하고 평가할 수 있습니다. 실행 시 에포크 수와 배치사이즈를 조절할 수 있습니다.

기본 실행
```bash
python main.py
```
에포크 수와 배치사이즈 조절
```bash
python main.py --epochs 10 --train_batch_size 16 --eval_batch_size 32
```

여러 GPU를 사용하여 분산 학습을 수행하려면 `torchrun`을 사용합니다.
```bash
torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE main.py --epochs 10 --train_batch_size 16 --eval_batch_size 32
```
--nproc_per_node: 한 노드에서 실행할 프로세스 수로, 일반적으로 사용할 GPU 수와 동일합니다.
--epochs, --train_batch_size, --eval_batch_size: 앞서 정의한 학습 관련 인자들입니다.
예를 들어, 4개의 GPU를 사용하고 싶다면:

```bash
torchrun --nproc_per_node=4 main.py --epochs 10 --train_batch_size 16 --eval_batch_size 32
```


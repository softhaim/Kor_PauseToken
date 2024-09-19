# Pause Token을 활용한 한국어 언어모델 Fine-tuning

이 프로젝트는 논문 [**"THINK BEFORE YOU SPEAK: TRAINING LANGUAGE MODELS WITH PAUSE TOKENS"**](https://arxiv.org/abs/2207.07696)을 기반으로, 한국어 언어모델에 Pause Token을 적용하여 Fine-tuning하는 코드입니다.

## 프로젝트 개요

언어모델의 학습 과정에서 Pause Token을 활용하여 모델의 성능을 향상시키는 방법을 연구합니다. 영어로 작성된 원 논문의 아이디어를 한국어에 맞게 적용하였으며, 한국어 접두사를 활용하여 Pause Token을 삽입하는 방식으로 전처리를 수행합니다.

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

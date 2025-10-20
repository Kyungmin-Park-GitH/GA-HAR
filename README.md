# GA-HAR

이 프로젝트는 8x8 적외선(IR) 기반 인간행동인식(HAR) 모델을 NSGA-II 기반 다목적 유전 알고리즘으로 최적화하는 예제 코드입니다. 두 개의 독립적인 데이터세트를 동시에 고려하며, 동일한 신경망 구조 및 하이퍼파라미터 구성을 각 데이터세트에 적용한 뒤 정확도를 계산하여 파레토 최적 개체를 탐색합니다.

## 주요 특징

- **데이터 준비**: CSV 형식의 40 프레임 × 8×8 IR 데이터를 로드하고, 테스트 인덱스 파일을 이용해 학습/테스트 세트를 분리한 뒤 각 프레임에 가우시안 스무딩을 적용하고 학습 세트 평균/표준편차로 정규화합니다.
- **모델 구조**: 기본 제공 LSTM 분류기는 각 프레임을 시계열로 펼쳐 층 수·유닛 수·드롭아웃을 탐색하며, LSTM 내부 활성화는 `tanh`·`sigmoid` 조합으로 고정되어 있습니다. 추가로 CNN 전용(모든 합성곱 층에 `ReLU` 적용)과 CNN+LSTM 하이브리드 모델(합성곱 `ReLU` + LSTM `tanh`/`sigmoid`)도 포함되어 있어 동일한 유전 알고리즘 파이프라인을 그대로 활용할 수 있습니다.
- **평가 전략**: 모든 세대·모든 개체에 대해 5-Fold 교차 검증과 조기 종료를 사용합니다. 각 폴드의 정확도/재현율/정밀도/F1-score를 기록하고, 데이터세트별 평균 정확도를 기반으로 `평균 정확도`와 `최소 정확도` 두 가지 목적 함수를 계산합니다.
- **세대별 로깅**: 터미널에는 '세대-개체-데이터세트-폴드' 진행 상황이 표시되며, 각 세대의 파레토 우수 50개체(랭크·군집 거리 순)가 확정된 후 폴드별 결과·혼동 행렬·평균 지표가 CSV에 순서대로 기록됩니다.
- **유전 알고리즘**: NSGA-II 절차를 따르며 세대당 50개의 개체를 유지하고 최대 100세대까지 탐색합니다. 각 하이퍼파라미터는 10% 확률로 변이되며, 자식은 3-토너먼트 선택과 50% 균등 교차로 생성됩니다.

## 하이퍼파라미터 검색 범위

- **공통값**
  - `learning_rate`: 0.0001부터 0.01까지 0.0001 간격
  - `dropout_rate`: 0.0부터 0.5까지 0.01 간격
  - `batch_size`: 8, 16, 32, 48, 64, 96, 128, 192, 256
- **CNN**
  - `conv_layers`: 1, 2, 3, 4, 5
  - `filters`: 1, 2, 4, 8, 16, 24, 32, 48, 64, 128
  - `kernel_size`: 1, 2, 3, 4, 5, 6, 7, 8
- **LSTM**
  - `lstm_layers`: 1, 2, 3, 4, 5
  - `units`: 1, 2, 4, 8, 16, 24, 32, 48, 64, 128
- **하이브리드**
  - 위 CNN과 LSTM 항목을 모두 포함해 동시에 탐색합니다.

## 실행 방법

```bash
python main.py \
    --coventry-path ./data/coventry_2018/40_linear_sensor1 \
    --coventry-test-indices diaz_coventry_108.txt \
    --infra-path ./data/infra_adl2018/40_sensor3 \
    --infra-test-indices diaz_infra_122.txt \
    --generations 100 \
    --population-size 50 \
    --output-dir results
```

인자로 다른 세대 수/개체 수를 전달하더라도 사양에 맞춰 최대 100세대, 세대당 50개체로 강제됩니다.

CUDA가 사용 가능한 환경에서는 자동으로 GPU를 사용합니다. 실행 로그에는 진행 중인 훈련/평가 단계와 최종 세대의 파레토 프론트에 속한 개체의 성능이 함께 출력됩니다. `--output-dir` 경로에는 다음과 같은 CSV가 생성됩니다.

- `<dataset>_details.csv`: 모든 세대·모든 폴드의 세부 결과(하이퍼파라미터 및 성능 지표).
- `<dataset>_confusion.csv`: 모든 세대·모든 폴드의 11×11 혼동 행렬.
- `overall_results.csv`: 각 개체의 평균 정확도/재현율/정밀도/F1-score(폴드 평균)와 두 목적 함수(`mean_accuracy_objective`, `min_accuracy_objective`).

## 피트니스 정의

각 데이터세트에 대해 5-Fold 검증으로 계산된 평균 정확도를 이용해 아래 두 개의 목적 함수를 구성합니다.

1. **mean_accuracy_objective** = (Coventry 평균 정확도 + Infra 평균 정확도) / 2
2. **min_accuracy_objective** = min( Coventry 평균 정확도, Infra 평균 정확도 )

NSGA-II는 위 두 값이 동시에 큰 개체를 선호하도록 파레토 지배 관계를 계산합니다. 따라서 두 데이터세트에서의 성능을 균형 있게 향상시키는 하이퍼파라미터 구성이 선택됩니다.

## 모델 파일 교체 안내

저장소에는 LSTM, CNN, CNN+LSTM 하이브리드 세 가지 구현이 동시에 들어 있습니다. 실행 전에 원하는 아키텍처에 맞춰 관련 파일 이름을 통일해야 합니다. 아래 단계는 **동시에 모두 수행**되어야 하며, 일부만 변경하면 임포트 오류가 발생합니다.

- **LSTM 실행 준비**
  1. `src/ga_lstm.py` → `src/ga.py`
  2. `src/training_lstm.py` → `src/training.py`
  3. `src/model_lstm.py` → `src/model.py`

- **CNN 실행 준비**
  1. `src/ga_cnn.py` → `src/ga.py`
  2. `src/training_cnn.py` → `src/training.py`
  3. `src/model_cnn.py` → `src/model.py`

- **하이브리드(CNN+LSTM) 실행 준비**
  1. `src/ga_hybrid.py` → `src/ga.py`
  2. `src/training_hybrid.py` → `src/training.py`
  3. `src/model_hybrid.py` → `src/model.py`

아키텍처를 변경하고 싶다면 위에서 변경한 파일명을 다시 원래 이름으로 되돌린 뒤, 원하는 아키텍처의 파일을 동일한 방식으로 `ga.py`, `training.py`, `model.py`로 맞춰 주세요. 코드 내부에서는 `ga.py`, `training.py`, `model.py`만을 참조하므로, 사용자가 파일명을 정확히 맞춰야 실행이 가능합니다.

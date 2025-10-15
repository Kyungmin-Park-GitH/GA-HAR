# GA-HAR

이 프로젝트는 8x8 적외선(IR) 기반 인간행동인식(HAR) 모델을 NSGA-II 기반 다목적 유전 알고리즘으로 최적화하는 예제 코드입니다. 두 개의 독립적인 데이터세트를 동시에 고려하며, 동일한 신경망 구조 및 하이퍼파라미터 구성을 각 데이터세트에 적용한 뒤 정확도를 계산하여 파레토 최적 개체를 탐색합니다.

## 주요 특징

- **데이터 준비**: CSV 형식의 40 프레임 × 8×8 IR 데이터를 로드하고, 테스트 인덱스 파일을 이용해 학습/테스트 세트를 분리합니다.
- **모델 구조**: 입력 40 프레임을 채널로 간주하여 2D 합성곱 신경망을 구성하고, 합성곱층에서 공간 해상도를 유지합니다.
- **평가 전략**: 초기 세대는 5-Fold 교차 검증과 조기 종료를 사용하여 평균 테스트 정확도를 계산하며, 이후 세대는 80/20 학습/검증 분할을 통해 단일 평가를 수행합니다. 모든 평가에서는 정확도, 재현율, 정밀도, F1-score, 혼동 행렬을 함께 저장합니다.
- **실시간 로깅**: 터미널에는 '세대-개체-데이터세트-폴드' 진행 상황이 표시되고, 초기 세대 폴드별 결과와 개체 통합 지표가 즉시 CSV로 저장됩니다.
- **유전 알고리즘**: NSGA-II 절차를 따르며 세대당 30개의 개체를 유지하고, 변이는 각 하이퍼파라미터에 대해 10% 확률로 적용합니다.

## 실행 방법

```bash
python main.py \
    --coventry-path ./data/coventry_2018/40_linear_sensor1 \
    --coventry-test-indices diaz_coventry_108.txt \
    --infra-path ./data/infra_adl2018/40_sensor3 \
    --infra-test-indices diaz_infra_122.txt \
    --generations 10 \
    --population-size 30 \
    --output-dir results
```

CUDA가 사용 가능한 환경에서는 자동으로 GPU를 사용합니다. 실행 로그에는 진행 중인 훈련/평가 단계와 최종 세대의 파레토 프론트에 속한 개체의 성능이 함께 출력됩니다. `--output-dir` 경로에는 다음과 같은 CSV가 생성됩니다.

- `initial_<dataset>_details.csv`: 초기 세대 5-Fold 결과(하이퍼파라미터 및 성능 지표).
- `initial_<dataset>_confusion.csv`: 초기 세대 각 폴드의 혼동 행렬.
- `overall_results.csv`: 모든 세대에서 평가된 개체의 통합 성능 지표.

## VS Code에서 실행하기

1. **Python 확장 설치**: VS Code의 확장(Extensions) 뷰에서 Microsoft의 "Python" 확장을 설치합니다. 필요하다면 "Pylance"도 함께 설치하여 코드 자동 완성과 정적 분석을 활용하세요.
2. **가상환경 준비**: VS Code의 터미널(`Ctrl + ``)을 열고 프로젝트 루트에서 아래 명령으로 가상환경을 생성 및 활성화합니다.

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell은 .venv\Scripts\Activate.ps1
   pip install -r requirements.txt  # 의존성 파일이 있다면 실행
   ```

   `requirements.txt`가 없다면 `pip install torch pandas scikit-learn numpy` 등 필요한 패키지를 수동으로 설치하세요.
3. **Python 인터프리터 선택**: VS Code 우측 하단의 인터프리터 선택기 또는 `Ctrl+Shift+P` → "Python: Select Interpreter"에서 `.venv`를 선택합니다.
4. **실행 구성**: VS Code의 `Run and Debug` 패널에서 "Add Configuration"을 선택하고 "Python File" 템플릿을 추가한 뒤 `main.py`가 실행되도록 경로를 지정합니다. 혹은 터미널에서 직접 `python main.py` 명령을 실행해도 됩니다.
5. **터미널 실행**: 통합 터미널에서 앞선 "실행 방법" 섹션의 명령을 그대로 실행하거나, `launch.json`을 구성했다면 `F5`로 디버깅을 시작합니다. 실행 중에는 진행 상황이 출력되고, 지정한 `--output-dir` 하위에 CSV 로그가 실시간으로 생성됩니다.

원격 SSH 환경에서 VS Code를 사용한다면, `Remote - SSH` 확장을 이용해 서버에 접속한 뒤 동일한 절차를 따르면 됩니다.

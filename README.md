# 다중 생체신호 이상 징후 탐지 프로토타입

PPG-DaLiA와 WESAD 형식의 웨어러블 데이터를 읽어 PPG/HRV, 움직임,
EDA, 피부온도 특징을 추출하고 개인별 baseline 대비 이상 정도를
`정상(normal) / 주의(caution) / 위험 징후(risk)`로 표시하는 교육·연구용 코드입니다.

> 이 프로젝트는 실신을 진단하거나 예측하는 의료기기가 아닙니다. 공개 데이터에는
> 실신 사건 라벨이 없으므로 현재 결과는 **평상시 대비 다중 생체신호 이상도**입니다.

## 1. 빠른 실행(실제 데이터 없이 확인)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_demo_data.py --output data/demo_subject.pkl
python scripts/prepare_dataset.py \
  --dataset ppg-dalia \
  --input data/demo_subject.pkl \
  --output data/features.csv \
  --window 60 --step 15
python scripts/train_prototype.py \
  --features data/features.csv \
  --output-dir outputs
```

생성 결과:

- `data/features.csv`: 슬라이딩 윈도우별 특징
- `outputs/scored_windows.csv`: 이상도와 단계
- `outputs/model.joblib`: 개인 baseline 이상탐지 모델
- `outputs/summary.json`: 단계별 개수와 설정

## 2. PPG-DaLiA 사용

PPG-DaLiA 원본은 약 2.7 GB입니다. 자동 다운로드 명령은 다음과 같습니다.

```bash
python scripts/download_data.py --dataset ppg-dalia --output-dir data/raw
```

압축을 푼 후 한 명의 pickle 파일을 지정합니다.

```bash
python scripts/prepare_dataset.py \
  --dataset ppg-dalia \
  --input data/raw/PPG_FieldStudy/S1/S1.pkl \
  --output data/ppg_dalia_s1_features.csv
```

전체 피험자 처리:

```bash
python scripts/prepare_dataset.py \
  --dataset ppg-dalia \
  --input 'data/raw/PPG_FieldStudy/S*/S*.pkl' \
  --output data/ppg_dalia_all_features.csv
```

## 3. WESAD 사용

WESAD 공식 페이지에서 파일을 받아 피험자별 `S2.pkl` 등의 파일을 지정합니다.

```bash
python scripts/prepare_dataset.py \
  --dataset wesad \
  --input 'data/raw/WESAD/S*/S*.pkl' \
  --output data/wesad_features.csv
```

WESAD의 실험 라벨은 중립/스트레스/즐거움 등이며 실신 라벨이 아닙니다.
본 코드는 중립 라벨(`1`)을 baseline 학습 후보로 사용할 수 있습니다.

## 4. 처리 흐름

1. 피험자별 다중 샘플링 주기 신호를 원본 해상도로 로드
2. 60초 윈도우, 15초 간격으로 신호 분할
3. PPG 대역통과 필터 및 peak 검출
4. HR, IBI, SDNN, RMSSD, pNN50 및 PPG 통계 특징 추출
5. 가속도 크기·jerk, EDA, 피부온도 특징 추출
6. 신호 품질이 낮은 윈도우 표시
7. 개인별 baseline 윈도우로 RobustScaler + IsolationForest 학습
8. 이상도를 0~1로 변환하여 단계 출력

기본 단계 경계값 0.45와 0.70은 임상적으로 검증된 수치가 아니라 UI와
파이프라인을 시험하기 위한 값입니다.

## 5. 자체 Galaxy Watch 데이터 연결

워치 앱에서 CSV를 내보낼 때 다음 필드를 권장합니다.

```text
subject_id,timestamp_ms,ppg,acc_x,acc_y,acc_z,eda,temp
```

원시 PPG와 센서별 샘플링 주기를 보존하는 것이 중요합니다. Galaxy Watch 접근
가능 항목과 샘플링 방식은 모델·지역·SDK 권한에 따라 달라질 수 있으므로 앱 구현
시점의 공식 SDK 문서를 다시 확인해야 합니다. CSV 어댑터는 다음 개발 단계에서
워치 앱의 실제 출력 형식에 맞춰 추가하면 됩니다.

## 6. 데이터 누수 방지

- 모델 성능평가 시 동일 피험자의 윈도우가 무작위로 학습·시험 세트에 섞이지 않도록
  피험자 단위로 분리해야 합니다.
- 공개 데이터에 실신 라벨이 없으므로 정확도·민감도·특이도를 실신 성능으로 보고하면
  안 됩니다.
- 자체 수집은 앉기, 서기, 걷기 등 안전한 정상 활동만 대상으로 하고 실신을 유발하지
  않습니다.

## 공식 데이터 출처

- PPG-DaLiA: https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia
- WESAD: https://archive.ics.uci.edu/dataset/465/wesad%2Bwearable%2Bstress%2Band%2Baffect%2Bdetection
- PhysioNet PTT-PPG: https://physionet.org/content/pulse-transit-time-ppg/1.1.0/
- PhysioNet wearable stress/exercise: https://physionet.org/content/wearable-device-dataset/1.0.1/


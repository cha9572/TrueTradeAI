# TrueTrade AI — Model 1

**TIGAS 선택 다운로드:** Python 3.14에서 Real/Fake 각 100장만 받는 명령과 각 2,000장으로 확대하는 방법은 [TIGAS_DOWNLOAD.md](TIGAS_DOWNLOAD.md)를 참고하세요. 자동 생성 CSV는 `data/tigas/metadata.csv`, 전용 학습 설정은 `configs/tigas_train.yaml`입니다. 일반 AI 이미지의 유형 미지정 Fake는 `fake_type=unknown`으로 지원하며 A/B/C로 임의 분류하지 않습니다.

이미지 **전체**를 입력받아 AI 생성·조작 의심 점수(0~1)를 출력하는 이진 분류 프로젝트입니다. EfficientNet-B0 → Global Average Pooling → 출력 1개, 학습은 `BCEWithLogitsLoss`, 추론은 sigmoid를 사용합니다. 이 점수는 검증·보정 전에는 실제 확률로 해석할 수 없습니다.

- Real=0, Fake A/B/C=1. A/B/C는 학습 클래스가 아니라 평가용 메타데이터입니다.
- Fake A: 메모만 AI inpainting, B: 상품 합성·교체, C: 전체 생성.
- 메모 탐지·메모 crop·OCR·segmentation·글씨 자연스러움 분석은 모델 2 범위이며 구현하지 않습니다.
- 종횡비를 유지하며 전체 이미지를 정사각형에 letterbox합니다. 중심 crop이나 랜덤 crop은 사용하지 않습니다. 학습에만 좌우 반전을 적용합니다.
- 384px로 축소하면 작은 국소 조작의 단서가 사라질 수 있습니다. A/B/C별 실제 데이터 평가가 필요합니다.

## 환경 설치

Python 3.11 권장. 이 컴퓨터는 Windows, RTX 4070 SUPER 12GB, NVIDIA driver 610.88입니다. `nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 버전이며 PyTorch의 CUDA runtime과 다를 수 있습니다.

Windows PowerShell (프로젝트 루트에서):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.check_gpu
```

Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m scripts.check_gpu
```

위 CUDA wheel 조합은 [PyTorch 공식 설치 목록](https://pytorch.org/get-started/previous-versions/)을 기준으로 선택했습니다. 다른 GPU/OS에서는 [공식 선택기](https://pytorch.org/get-started/locally/)로 호환 조합을 확인하세요. CPU만 사용하려면 첫 설치 명령의 index를 `https://download.pytorch.org/whl/cpu`로 바꾸세요. 시스템 전체 CUDA Toolkit 설치는 일반적인 wheel 실행에 필요하지 않습니다.

이후 Windows 명령의 `.\.venv\Scripts\python.exe`는 Linux에서 활성화된 환경의 `python`으로 바꿔 실행합니다. PowerShell 가상환경 활성화 없이도 모든 명령을 실행할 수 있습니다. `requirements-lock.txt`에는 이번 Windows/CUDA 검증 환경의 정확한 패키지 버전을 기록했습니다. 동일 환경을 재구성할 때 CUDA wheel 설치 후 `-r requirements-lock.txt`로 설치할 수 있습니다. 다른 OS에서는 호환 wheel을 먼저 확인하세요.

`check_gpu`는 필수 패키지 import, 실제 CUDA tensor 연산, GPU 이름, PyTorch CUDA 빌드, 드라이버를 확인합니다. CUDA가 없으면 CPU 빌드인지 초기화 문제인지 출력합니다. `training.device: auto`는 CUDA 우선, 불가능하면 이유와 CPU 학습이 느리다는 경고를 출력합니다. `cuda`를 지정했는데 사용할 수 없으면 오류로 종료합니다.

## 실제 데이터 입력

`data/metadata.csv`는 헤더만 제공됩니다. 실제 이미지를 준비한 후 행을 추가하세요. 이미지를 확보하지 않은 상태에서 실제 학습 결과를 만들지 않습니다.

```csv
image_path,label,fake_type,source_id,split,domain,photographer_id,item_id
data/raw/real001.jpg,0,real,scene001,train,trade,user001,item001
data/processed/fakeA001.png,1,A,scene001,train,trade,user001,item001
data/processed/fakeB001.png,1,B,scene001,train,trade,user001,item001
data/processed/fakeC001.png,1,C,generated001,train,trade,,
```

위 경로는 형식 예시입니다. 실제 파일은 별도로 넣어야 합니다. 필수 컬럼은 `image_path,label,fake_type,source_id,split`, split은 `train`, `val`, `test`입니다. 세 split 모두 준비하세요. 같은 원본 및 파생 Real/Fake A/B는 같은 `source_id`와 split을 사용합니다. 그룹별로 먼저 split을 정한 뒤 파생 이미지를 배치하세요. 임의의 행 단위 분할을 수행하지 않습니다.

상대 이미지 경로는 `data.root` 기준입니다. 설정 파일은 프로젝트의 `configs/` 아래에 두며, `data.root`, `data.metadata`, `training.output_dir`는 프로젝트 루트 기준입니다. 절대 경로도 사용할 수 있습니다. 선택 컬럼 `photographer_id`, `item_id`가 있으면 누수를 검사하며, 다른 그룹 컬럼도 `data.leakage_columns`에 추가할 수 있습니다. 빈 선택 ID는 검사에서 제외합니다. 잘못된 라벨/type, 빈 필수 ID, split 간 그룹 중복, 중복 이미지 경로, 누락·손상 이미지에서 실패합니다. 별도 파일로 복제한 동일 이미지는 경로 검사로 발견할 수 없으므로 원본 ID를 정확히 관리해야 합니다.

선택 `domain` 컬럼에 `general`(일반 AI 이미지), `trade`(중고거래 인증샷) 등을 넣으면 동일 임계값으로 domain별 지표도 저장합니다. 이 컬럼이 없으면 전체 성능만 계산합니다. 일반 이미지 결과로 인증샷 성능을 대신 주장하지 마세요.

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_dataset --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m src.train --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m src.predict --image data/sample.jpg --checkpoint outputs/checkpoints/best_model.pth
```

Linux:

```bash
python -m scripts.validate_dataset --config configs/baseline.yaml
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --config configs/baseline.yaml
python -m src.predict --image data/sample.jpg --checkpoint outputs/checkpoints/best_model.pth
```

추론 출력은 `Fake Probability: ...`, `Prediction: Real/Fake`입니다. 체크포인트가 없으면 명확한 오류를 출력합니다. 신뢰할 수 있는 체크포인트만 로드하세요.

## 설정과 평가 원칙

`configs/baseline.yaml`: ImageNet 사전학습 EfficientNet-B0, 384×384, AdamW lr=1e-4, batch=16, 20 epochs, cosine scheduler, CUDA AMP, seed=42, patience=5. 메모리 부족 시 `training.batch_size`를 줄이세요. AMP overflow로 갱신이 모두 생략되면 실패로 종료합니다. 이때 `training.amp_init_scale`을 낮추거나 `amp: false`로 재실행하세요. 로그의 `optimizer_updates`에 실제 갱신 수를 기록합니다. 검증·평가·추론은 float32를 사용합니다. 작은 데이터에서도 training batch가 1개 이미지가 되지 않게 구성하며 마지막 singleton batch는 버립니다. Windows 기본 `num_workers=0`; 필요에 따라 늘릴 수 있습니다. 재현성 seed를 고정하지만 장치·라이브러리 버전 간 비트 단위 일치를 보장하지는 않습니다.

사전학습 가중치는 첫 실제 학습에서 인터넷 다운로드가 필요합니다. 다운로드 실패 시 무작위 가중치로 자동 대체하지 않습니다. [timm 공식 API](https://huggingface.co/docs/timm/main/reference/models)의 모델 이름과 pretrained 옵션을 사용합니다. ResNet/ViT로 교체하려면 `model.name`과 해당 모델의 입력 크기·mean/std 등을 함께 검토하세요. 기본 전처리는 EfficientNet ImageNet 기준입니다.

매 epoch **validation에서만** F1 최대 임계값을 선택하고 해당 F1로 best checkpoint를 선택합니다. 동률이면 0.5에 가까운 임계값을 택합니다. `tune_threshold: false`면 설정의 고정 threshold를 저장합니다. 임계값 탐색 시 validation에는 두 클래스가 모두 있어야 합니다. test loader는 학습에서 만들지 않습니다. test에서는 checkpoint의 임계값과 전처리를 고정하며 재탐색 옵션을 제공하지 않습니다. validation 반복 활용의 과적합을 피하려면 충분한 검증 데이터가 필요합니다.

`best_model.pth`, `last_model.pth`는 모델/optimizer/scheduler/scaler 상태, epoch, best F1, 임계값, 설정, smoke 표시, seed·환경 정보를 저장합니다. `last`는 마지막 epoch, `best`는 validation F1 최고 epoch입니다. 학습 CLI는 새 실행이며 resume 기능은 포함하지 않습니다. 기존 checkpoint가 있는 출력 폴더의 덮어쓰기를 막으므로 실험마다 새 `output_dir`를 지정하세요.

평가 결과:

- `outputs/predictions/test_metrics.json` 및 `test_metrics.csv`: Accuracy, Precision, Recall, F1, ROC-AUC, 2×2 confusion matrix (행=true, 열=prediction, 순서 Real/Fake), A/B/C recall 및 표본 수, domain별 지표.
- `outputs/predictions/test_predictions.csv`: `image_path,label,fake_type,fake_probability,prediction,source_id` (+ domain). prediction은 0/1로 모델 2와 결합 가능합니다.
- 클래스가 하나뿐이면 ROC-AUC는 JSON `null`/CSV 빈 값, 없는 Fake 유형 recall도 `null`로 저장합니다. Precision/Recall/F1의 0분모는 0으로 처리합니다.
- `outputs/logs/training.csv`, `outputs/figures/training_curves.png`, `outputs/config.yaml`, metadata 사본 및 환경 정보를 함께 저장합니다.

test 평가에 다른 metadata를 넣더라도 학습/검증에서 사용한 source, 선택 그룹 ID, 이미지 경로와 겹치면 거부합니다. 학습 당시 기록한 identity와 비교하므로 test 파일의 split만 바꾸는 방식으로 우회할 수 없습니다. 학습 때 사용한 선택 ID 컬럼을 새 평가 metadata에서 누락해도 검사 불가 오류로 종료합니다. ID가 다른 복제 이미지나 의미적으로 같은 장면까지 자동 탐지하지는 않습니다.

## 합성 데이터로 코드 동작 확인

아래 데이터는 도형과 문자열로 만든 **SYNTHETIC ONLY** fixture입니다. AI 생성·조작 데이터셋이 아니며 모든 출력 점수와 지표는 파이프라인 확인용입니다. 모델 성능으로 인용하면 안 됩니다. `configs/smoke.yaml`은 인터넷 가중치 없이 64px, 1 epoch로 실행합니다. baseline과 데이터·출력·체크포인트를 분리합니다. smoke checkpoint를 평가·추론하려면 명시적인 `--allow-smoke`가 필요합니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.create_smoke_test_data
.\.venv\Scripts\python.exe -m scripts.validate_dataset --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m src.evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoints/best_model.pth --allow-smoke
.\.venv\Scripts\python.exe -m src.predict --image data/smoke/smoke_test_0_A.png --checkpoint outputs/smoke/checkpoints/best_model.pth --allow-smoke
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Linux에서는 같은 명령의 실행기를 `python`으로 바꾸세요. 생성기는 기존 `data/smoke/metadata.csv`를 덮어쓰지 않습니다.

## 파일 안내

| 경로 | 역할 |
|---|---|
| `configs/baseline.yaml`, `configs/smoke.yaml` | 실제 실험 기본값 / 합성 검증 전용 설정 |
| `requirements.txt`, `requirements-lock.txt` | 호환 의존성 범위 / 이번 검증 환경의 정확한 버전 |
| `src/dataset.py` | CSV 검증, 누수 검사, 전체 이미지 전처리, Dataset |
| `src/model.py` | timm backbone과 단일 logit head |
| `src/train.py` | train/validation, AMP, best/last 저장, early stopping, 곡선 |
| `src/evaluate.py` | 저장된 임계값으로 test 평가 및 개별 예측 저장 |
| `src/predict.py` | 전체 이미지 한 장 추론 |
| `src/utils.py` | config, seed, 장치, 지표, 임계값, 체크포인트 공통 처리 |
| `scripts/check_gpu.py` | OS/Python/import/CUDA 진단 |
| `scripts/check_model.py`, `scripts/check_checkpoint.py` | 384px forward / 학습 가중치 변경 및 체크포인트 복원 검사 |
| `scripts/validate_dataset.py` | 학습 전 전체 데이터 검증과 분포 출력 |
| `scripts/create_smoke_test_data.py` | 실제 성능 평가에 쓰지 않는 합성 fixture 생성 |
| `tests/` | 데이터 누수, 단일 클래스 지표, 임계값 및 checkpoint 회귀 검사 |
| `data/raw`, `processed`, `splits` | 원본·가공 이미지·분할 자료 보관 |
| `outputs/` | 실제 실험 산출물; `outputs/smoke/`는 합성 검증 전용 |
| `VALIDATION.md` | 이번 구현의 실제 실행 결과와 남은 검증 항목 |

실제 성능은 실제 데이터의 출처·생성기·JPEG 압축·촬영자·상품 분포에 크게 영향을 받습니다. 다음 실험에서는 충분한 A/B/C 표본, 생성기와 출처를 분리한 외부 테스트, 압축·리사이즈 강건성, 확률 보정과 운영 임계값 검증을 진행하세요.

# TIGAS 선택 다운로드

전체 저장소를 받지 않고 Hugging Face Hub의 `hf_hub_download`에 **이미지별 filename**을 지정합니다. `snapshot_download`, `load_dataset`, git clone, 대형 archive/parquet 다운로드는 사용하지 않습니다. 먼저 세 annotation CSV(합계 약 7.4 MB)를 받은 뒤 선택한 이미지 파일만 받습니다.

## 실제 확인한 원격 구조

저장소: [H1merka/TIGAS_dataset](https://huggingface.co/datasets/H1merka/TIGAS_dataset)

고정 revision: `6005716de6e91a243090987e2017af9e0b8565e7`

```text
train/annotations01.csv
train/images/<generator>/0_real|1_fake/<filename>
val/annotations01.csv
val/images/<generator>/0_real|1_fake/<filename>
test/annotations01.csv
test/images/<generator>/0_real|1_fake/<filename>
```

데이터셋 카드에는 이미지가 `train/images`에 공통 저장된다는 설명이 있지만, **실제 고정 revision의 Hub tree에는 각 split별 이미지 폴더가 있습니다.** 코드는 실제 tree와 annotation을 기준으로 확인하며 test 이미지를 train 경로로 잘못 요청하지 않습니다. CSV 원본의 Windows `\` 구분자를 `/`로 정규화합니다.

실제 annotation 집계:

| upstream split | Real=0 | Fake=1 | 합계 |
|---|---:|---:|---:|
| train | 70,663 | 70,627 | 141,290 |
| val | 7,078 | 7,089 | 14,167 |
| test | 7,038 | 7,089 | 14,127 |

세 annotation 사이의 정규화된 상대 이미지 경로 겹침은 0건이었습니다. 이 숫자는 annotation 행 집계이며 저장소의 모든 이미지가 다운로드 가능함을 주장하지 않습니다. 선택한 각 경로는 별도로 Hub 파일 정보와 실제 파일을 검사합니다. 상세 tree와 집계는 `data/tigas/repository_structure.json`에 기록합니다.

## Windows / Python 3.14 실행

현재 작업 폴더에는 Python 3.14용 `.venv-download`를 준비했습니다. 다운로드에는 PyTorch, torchvision, pandas, datasets가 필요하지 않습니다.

```powershell
cd "C:\Users\차기원\Desktop\컴퓨터비전"

# 최초 기본값: Real 100장 + Fake 100장
.\.venv-download\Scripts\python.exe -m scripts.download_tigas

# 다운로드 없이 annotation 및 실제 shallow tree만 점검
.\.venv-download\Scripts\python.exe -m scripts.download_tigas --inspect-only

# 이미지는 다시 받지 않고 샘플 10장 그림만 생성
.\.venv-download\Scripts\python.exe -m scripts.visualize_tigas_samples
```

새 PC라면 먼저 전용 환경을 구성합니다.

```powershell
py -3.14 -m venv .venv-download
.\.venv-download\Scripts\python.exe -m pip install -r requirements-download.txt
```

Linux에서는 `python3.14 -m venv .venv-download` 후 `.venv-download/bin/python -m pip install -r requirements-download.txt` 및 `.venv-download/bin/python -m scripts.download_tigas`를 사용합니다. Linux 실제 실행 검증은 하지 않았습니다.

## 2,000장씩으로 확대

`configs/tigas_download.json`의 `per_class`를 `100`에서 `2000`으로 바꾸면 됩니다. 설정 파일을 바꾸지 않고 이번 실행만 확대할 수도 있습니다.

```powershell
.\.venv-download\Scripts\python.exe -m scripts.download_tigas --per-class 2000
```

| per_class | train Real/Fake 각각 | val 각각 | test 각각 | 총 이미지 |
|---:|---:|---:|---:|---:|
| 100 | 80 | 10 | 10 | 200 |
| 2000 | 1600 | 200 | 200 | 4000 |

기존 upstream train/val/test를 그대로 유지합니다. train 이미지를 새 test로 나누거나 기존 test를 train에 섞지 않습니다. seed와 경로의 SHA256 순위로 안정적으로 선택하므로 요청 수를 늘리면 기존 선택의 prefix를 재사용합니다. seed나 revision을 바꾸려면 별도의 `output_dir`를 사용해야 합니다. 이 구현은 `per_class <= 2000`, 이미지 하나당 기본 25 MiB, 이미지 전송 예약량 기본 최대 4096 MiB의 상한을 둡니다. annotation 전송량과 프로토콜 overhead는 이미지 전송 예약량과 별도입니다. 한도 부족 시 전체 다운로드로 전환하지 않고 오류로 종료합니다.

## 파일 재사용과 실패 기록

각 선택 파일에 대해 다음을 확인합니다.

1. 정확한 원격 경로가 고정 revision에 존재하는지와 원격 크기 확인.
2. 로컬 파일의 실제 크기와 LFS SHA256 또는 Git blob hash 비교.
3. Pillow `verify()`와 전체 이미지 디코딩 확인.
4. 같은 콘텐츠가 다른 split/라벨에 이미 사용됐다면 건너뛰고 대체 후보 선택.

검사를 통과한 로컬 파일은 `reused`로 기록하고 이미지 다운로드 함수를 호출하지 않습니다. 손상된 파일은 `invalid_local`로 기록하고 해당 파일만 다시 받습니다. 다운로드 실패는 기본 두 번 재시도하며 최종 실패 시 같은 upstream split·라벨의 다음 후보로 대체합니다. 10개 파일이 최종 실패하면 중단합니다. 기본 순차 다운로드이므로 수천 장은 네트워크 상태에 따라 시간이 걸립니다. 두 터미널의 동시 실행은 lock으로 막습니다.

`download_log.csv`에는 성공(`downloaded`), 재사용(`reused`), 실패 시도(`failed_attempt`), 최종 실패(`failed`), 중복 차단(`skipped_duplicate`), 실행 중단(`run_aborted`) 및 오류 사유를 append합니다. 성공 콘텐츠의 split 소유권을 로그에서 복원하므로 재실행·확대·축소 후에도 기존 test 콘텐츠를 새 train에 넣지 않습니다. 바이트가 같은 복제만 검사하며 재압축·근접 중복의 의미적 동일성은 보장하지 않습니다.

정상 완료 시 `metadata.csv`를 원자적으로 갱신합니다. 실패·중단 시에는 성공한 부분만 `metadata.partial.csv`에 기록하고 이전 정상 `metadata.csv`는 보존하며 exit 1을 반환합니다. 중단된 실행은 같은 명령으로 다시 실행하세요. 파일은 재사용됩니다.

## 생성되는 파일

```text
data/tigas/
  hub/train|val|test/annotations01.csv
  hub/train|val|test/images/...       # 선택한 이미지만 저장
  metadata.csv                      # 학습용 완성 CSV
  download_log.csv                  # 모든 실행의 append-only 결과
  download_summary.json             # 최신 실행 결과 및 집계
  repository_structure.json         # 실제 shallow tree와 annotation 수
  dataset_state.json                # repo/revision/seed 고정
  runs/<run_id>.json                # 실행별 결과
outputs/tigas/
  samples.png                       # Real 5 + Fake 5 그림
  samples.csv                       # 선택된 10장의 라벨·경로 확인표
```

기존 `data/metadata.csv`, baseline 설정, smoke 데이터와 체크포인트는 덮어쓰지 않습니다.

학습용 CSV는 기존 필수 컬럼과 provenance를 포함합니다.

```text
image_path,label,fake_type,source_id,split,domain,dataset,upstream_split,repo_path,revision,generator,sha256,width,height
```

`label`은 원본의 Real=0/Fake=1을 그대로 유지합니다. **TIGAS에는 TrueTrade의 Fake A/B/C 주석이 없으므로 Fake는 `fake_type=unknown`**으로 저장합니다. 이를 전부 C 또는 다른 A/B/C로 가정하지 않습니다. 기존 Dataset에 `unknown` 지원을 추가했으며 이 역시 이진 Fake=1로 학습합니다. A/B/C별 Recall은 해당 유형이 없으면 `null`입니다. `domain=general`로 저장하므로 중고거래 인증샷 결과와 구분합니다. `source_id`는 이미지 콘텐츠 SHA256 기반 ID이며 TIGAS에 없는 원본/파생 관계를 추정하지 않습니다.

## 10장 시각화

다운로드 정상 완료 후 `visualize_tigas_samples.py`를 자동 실행합니다. Real 5장과 Fake 5장을 고르고 원본 split annotation의 라벨을 다시 대조하며 실제 로컬 파일을 열어 확인합니다. 그림에는 라벨·원본 split·로컬 경로를 표시하고 별도 CSV에 10개 원격/로컬 경로를 보존합니다. 이것은 주석과 파일의 일치 검사이며 이미지의 진위를 사람 대신 재판정하는 검사는 아닙니다.

## 모델 1 학습에 사용

다운로더는 Python 3.14에서 실행하지만 학습은 이미 검증한 Python 3.11 `.venv`를 사용할 수 있습니다. 다운로드 때문에 기존 학습 환경을 3.14로 바꾸지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_dataset --config configs/tigas_train.yaml
.\.venv\Scripts\python.exe -m src.train --config configs/tigas_train.yaml
.\.venv\Scripts\python.exe -m src.evaluate --config configs/tigas_train.yaml --checkpoint outputs/tigas_training/checkpoints/best_model.pth
```

이 작업에서는 다운로드와 학습 연결을 검증하며 실제 학습을 자동 시작하지 않습니다. Real/Fake 각 100장은 실행 확인용 소규모 실제 데이터입니다. TIGAS 결과를 TrueTrade 인증샷 A/B/C 탐지 성능으로 해석하면 안 됩니다.

## 검증 명령

```powershell
# Python 3.14: 다운로드 기능만 검사, 네트워크 없이 실행
.\.venv-download\Scripts\python.exe -m unittest tests.test_tigas_download -v

# 기존 학습 환경 전체 회귀 검사
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Hub API 사용법은 [Hugging Face 공식 다운로드 문서](https://huggingface.co/docs/huggingface_hub/guides/download)를 참고했습니다. 이미지별 사용 조건은 [데이터셋 라이선스 안내](https://huggingface.co/datasets/H1merka/TIGAS_dataset#license)를 확인하세요.

## 이번 실행의 실제 검증 결과

Windows / Python **3.14.7** 다운로드 환경에서 실행했습니다.

| 검사 | 결과 |
|---|---|
| 첫 다운로드 | Real 100 + Fake 100, 성공 200, 실패 0 |
| 이미지 파일 합계 | 82,344,504 bytes (약 82.3 MB / 78.5 MiB) |
| 분할 | train 160, val 20, test 20; 각 split Real/Fake 동수 |
| 재실행 | 200장 모두 reused; 이미지 전송 예약량 0 bytes |
| 실제 파일 | 선택한 200개 파일만 존재, 모두 검증 통과 |
| 원본 라벨·분할 | 전체 200행을 원본 annotation과 대조하여 일치 확인 |
| 콘텐츠 누수 | 선택된 파일의 동일 SHA256이 여러 split에 나타나지 않음 |
| 샘플 10장 | Real 5/Fake 5의 라벨·경로 대조 및 PNG 시각 검토 완료 |
| 기존 학습 코드 연결 | Python 3.11에서 세 split 모두 `(2,3,384,384)` 배치 로딩 통과 |
| 다운로드 단위·통합 테스트 | Python 3.14에서 10개 통과 |
| 프로젝트 전체 회귀 테스트 | Python 3.11에서 31개 통과 |

실패·손상·재시도·byte budget·중복 차단은 임시 파일과 모의 Hub API로 테스트했습니다. 실제 다운로드에서는 실패가 없었습니다. 각 2,000장 설정의 split 수 계산과 안정적인 선택 순위를 검사했지만 **4,000장 다운로드 자체는 실행하지 않았습니다.** 이번 요청에 따라 첫 실행은 200장으로 제한했습니다. 실제 학습이나 탐지 성능 평가는 실행하지 않았습니다.

실행별 상세 결과는 `data/tigas/runs/`, 종합 확인표는 `data/tigas/verification.json`, 전체 로그는 `data/tigas/download_log.csv`에 있습니다.

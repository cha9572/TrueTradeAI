# 구현 및 검증 기록 — 2026-11-09

모델 1은 전체 이미지 기반 Real/Fake 이진 분류입니다. A/B/C는 평가용 메타데이터이며 메모 탐지, crop, OCR, segmentation, 손글씨 분석 기능은 포함하지 않습니다.

## 확인한 환경

| 항목 | 실제 확인 결과 |
|---|---|
| OS | Windows, build 26200 (Python 플랫폼 문자열: Windows-10-10.0.26200-SP0) |
| Python | 전역 3.14와 별도로 3.11.9 설치, 프로젝트 `.venv` 구성 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12282 MiB |
| Driver | 610.88 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| CUDA 사용 및 실제 연산 | `torch.cuda.is_available() == True`, CUDA tensor 연산 성공 |
| 주요 패키지 | timm 1.0.31, sklearn 1.9.1, pandas 2.3.3, Pillow 12.3.0 |
| 전체 버전 | `requirements-lock.txt`, `outputs/smoke/environment.json` |

## Phase별 실행 결과

1. **Phase 1 통과**: 빈 작업 폴더 확인, 가상환경 및 패키지 설치, CSV/Dataset 구현. 합성 24장(train/val/test 각 8장, 각 split의 real/A/B/C 각 2장) 검증 및 로딩. 데이터 누수·경로·라벨·type·손상 이미지·전체 이미지 유지 검사 통과.
2. **Phase 2 통과**: CUDA에서 EfficientNet-B0의 `(2,3,384,384) → (2,)` finite logit 확인. 무작위 초기화와 실제 ImageNet 사전학습 가중치 **두 경로** 모두 forward 성공. `pretrained=True` 가중치가 프로젝트 cache에 다운로드됨.
3. **Phase 3 통과**: 합성 전용 64px 설정으로 CUDA AMP 1 epoch, batch=4 실행. 실제 optimizer 갱신 **2회**, backbone 파라미터 변경 확인. best/last 저장·복원, float32 검증, CSV 로그, loss/F1 곡선 저장 및 그림 확인. 저장된 epoch=1, threshold=0.4285796284675598은 **합성 파이프라인 검증 전용** 값.
4. **Phase 4 통과**: test 8장 GPU 평가, Accuracy/Precision/Recall/F1/AUC/confusion matrix/A·B·C recall 및 `domain=smoke` 지표 JSON/CSV 생성. 개별 이미지 예측 CSV 생성. GPU 및 CPU 단일 이미지 추론 확인. CPU 단일 결과와 GPU 배치 결과의 점수 차이 <1e-5 확인.

합성 샘플은 AI 조작 데이터가 아닙니다. 위 숫자·체크포인트는 코드 검증만 의미하며 실제 모델 탐지 성능을 나타내지 않습니다. 실제 성능 수치로 보고하지 않았습니다.

## 추가 검증

- `python -m unittest discover -s tests -v`: **20개 통과**. Windows spawn 방식의 `num_workers=1` 로더 포함.
- `python -m compileall -q src scripts tests`: 통과.
- `python -m pip check`: 의존성 충돌 없음.
- test 설정의 threshold를 0.99999로 바꿔도 checkpoint의 저장된 threshold 유지 확인.
- test에 Real만 남긴 **평가 CLI 내부 경로**까지 실행: ROC-AUC와 부재 Fake 유형 Recall이 JSON `null`로 정상 저장됨.
- 학습·검증 당시 ID와 평가 ID를 비교하는 보호 로직 통과. 선택 ID 컬럼을 제거해서 검사를 회피하는 경우도 거부.
- 없는 체크포인트 예측: exit 1 및 학습 필요 오류 확인.
- 합성 checkpoint에 `--allow-smoke` 미지정: exit 1 확인.
- 헤더만 있는 실제 metadata: 빈 데이터 오류와 비정상 종료 확인.
- 이미지별 prediction이 `fake_probability >= checkpoint.threshold`와 모두 일치함.

통합 검증 세부 결과는 `outputs/smoke/verification.json`, 학습 산출물은 `outputs/smoke/`에 있습니다. 실제 실험의 기본 출력 폴더에 학습된 모델이 있는 것처럼 만들지 않았습니다.

## 검증 중 발견해 수정한 문제

- 높은 AMP 초기 스케일에서 모든 batch의 optimizer 갱신이 생략되는 현상: 초기 스케일을 설정으로 노출, 갱신 횟수를 기록하고 0회면 실패 처리. 낮은 스케일로 재실행해 실제 weight 변경 확인.
- PyTorch 버전 객체가 checkpoint에 직렬화되어 `weights_only=True` 복원이 거부되는 현상: 환경 버전을 일반 문자열로 저장하도록 수정 후 재학습·복원 검증. 안전 로딩을 해제하지 않음.
- Windows 다중 worker에서 lambda 전처리가 pickle되지 않는 문제: lambda 제거 후 spawn 로더 테스트 통과.
- 최초 package/가중치 다운로드가 sandbox 네트워크 제한으로 실패: 승인된 다운로드로 재실행 후 성공. matplotlib cache는 쓰기 가능한 프로젝트 내부에 둠.

## 실행하지 않은 항목과 이유

- **실제 데이터 학습 및 A/B/C 탐지 성능 평가**: 실제 데이터가 제공되지 않았음. `data/metadata.csv`는 의도적으로 헤더만 있음.
- **384px, 20 epoch 전체 학습**: 실제 데이터가 없으므로 실행하지 않음. 384px 사전학습 모델 forward와 64px 1 epoch 파이프라인을 각각 검증함. 실험 batch=16의 최대 GPU 메모리 사용량은 실데이터에서 확인 필요.
- **Linux 실행**: 현재 호스트는 Windows. Linux 명령과 플랫폼 독립 코드를 제공했지만 Linux에서 실행했다고 주장하지 않음.
- **장기간 early stopping 동작 실험**: 1 epoch smoke이므로 patience=5 발동까지 실행하지 않음. 학습 로직은 구현되어 있음.

## 사용 및 다음 작업

전체 파일 역할·설치·학습·평가·추론 명령은 `README.md`에 정리했습니다. 실제 JPG/PNG 이미지와 그룹 단위로 분리한 metadata를 추가하고 `scripts.validate_dataset`를 먼저 실행하세요. baseline은 ImageNet 사전학습을 사용합니다. 생성기·촬영자·상품 분포를 분리한 실제 평가, A/B/C별 충분한 표본, 압축·리사이즈 강건성과 점수 보정을 후속으로 확인해야 합니다.

## 추가 작업: TIGAS 선택 다운로드

이후 요청에 따라 Python 3.14.7 전용 다운로드 환경에서 TIGAS Real/Fake 각 100장(총 200장)을 실제로 받았습니다. 이미지 합계 82,344,504 bytes이며 기존 upstream train/val/test 분할을 그대로 유지합니다. 재실행 시 200장 모두 재사용되어 이미지 전송이 발생하지 않았습니다. 자동 CSV는 `data/tigas/metadata.csv`, 전용 학습 설정은 `configs/tigas_train.yaml`입니다. 기존 헤더 전용 `data/metadata.csv`는 그대로입니다.

TIGAS는 일반 영역의 이진 라벨만 제공하므로 `fake_type=unknown`, `domain=general`로 저장합니다. TrueTrade A/B/C로 임의 지정하지 않습니다. 원본 라벨과 split, 실제 이미지 경로 및 해시 검사를 수행했고 Real 5/Fake 5 샘플 그림을 생성·검토했습니다. 실제 TIGAS 세 split의 384px 전체 이미지 배치 로딩도 통과했습니다. 이 데이터로 모델 학습은 아직 실행하지 않았으며 TrueTrade 인증샷 데이터도 여전히 별도 준비가 필요합니다.

다운로드 관련 테스트는 Python 3.14에서 10개, 기존 프로젝트 전체 회귀는 Python 3.11에서 31개가 통과했습니다. 상세 사용법·실제 원격 구조·추가 검증 결과는 `TIGAS_DOWNLOAD.md`, 기계 확인표는 `data/tigas/verification.json`을 참고하세요. 각 2,000장 다운로드는 설정과 quota 계산만 검증했으며 실제 4,000장 다운로드는 실행하지 않았습니다.

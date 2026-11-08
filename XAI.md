# 모델 1 Grad-CAM 설명

기존 EfficientNet-B0와 학습된 checkpoint를 그대로 사용합니다. 추가 학습이나 외부 XAI 패키지 설치 없이 PyTorch autograd로 Fake logit의 Grad-CAM을 계산합니다. 기존 학습·평가 코드는 변경하지 않았습니다.

## 실행

기존 예측 명령에 `--explain`을 추가하세요. 아래는 현재 다운로드한 TIGAS 샘플과 학습한 checkpoint를 사용하는 예시입니다. 다른 이미지를 설명하려면 `--image` 경로를 바꾸세요.

```powershell
.\.venv\Scripts\python.exe -m src.predict --image data/tigas/hub/test/images/DALLE2/1_fake/0220.png --checkpoint outputs/tigas_training/checkpoints/best_model.pth --explain
```

저장 폴더를 직접 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m src.predict --image data/tigas/hub/test/images/DALLE2/1_fake/0220.png --checkpoint outputs/tigas_training/checkpoints/best_model.pth --explain --explain-dir outputs/explanations/tigas_example
```

Linux에서는 활성화된 학습 환경의 `python -m src.predict ...`로 실행합니다. `--explain`이 없으면 기존 Fake Probability/Prediction 출력과 반환값을 유지합니다. 저장된 임계값도 그대로 사용합니다. smoke checkpoint에는 기존과 동일하게 `--allow-smoke`가 필요하며 결과에도 smoke 표시를 남깁니다.

기본 결과 폴더는 `outputs/explanations/<이미지이름>_<이미지해시>_<checkpoint해시>/`입니다. 같은 이미지와 checkpoint를 다시 설명하면 같은 결과 폴더를 사용합니다. 이미지와 모델이 달라지면 해시가 달라져 서로 다른 폴더를 사용합니다. 모든 결과는 기존 `.gitignore`의 `outputs/` 규칙으로 Git 업로드에서 제외됩니다.

## 저장 파일

| 파일 | 내용 |
|---|---|
| `original.png` | EXIF 방향을 적용한 RGB 원본 |
| `heatmap.png` | 원본 크기로 정렬한 색상 heatmap |
| `overlay.png` | 원본 위에 heatmap을 겹친 결과 |
| `explanation.png` | 원본·heatmap·overlay, Fake Probability·판정·임계값을 함께 표시 |
| `heatmap_values.npy` | 원본 좌표의 float32 상대 중요도, 0~1 |
| `explanation.txt` | 측정된 위치에 근거한 한국어 설명과 해석 한계 |
| `explanation.json` | 확률·임계값·계층·gradient·좌표·3×3 집계·해시·파일 경로 |

그림은 폰트 호환성을 위해 영어로 표시하며 한국어 설명은 터미널과 TXT/JSON에 제공합니다. 이미지에 아무 응답이 없는 경우에는 0 map을 저장하고 강조 위치를 만들어내지 않습니다. 0 응답 픽셀은 overlay에서 원본 색을 그대로 유지합니다.

## 계산 대상과 좌표

- 계층: `network.conv_head`, timm EfficientNet-B0의 마지막 `Conv2d(320,1280,kernel_size=1)`.
- 위치: 마지막 BN/activation 및 global average pooling 이전. 384×384 입력의 feature map은 `(1,1280,12,12)`입니다.
- 대상: **Fake logit**. sigmoid 확률 대신 logit의 gradient를 사용하며 sigmoid 출력 1개를 두 클래스 중 index 1로 잘못 처리하지 않습니다.
- 수식: 채널별 gradient를 공간 평균한 가중치와 activation의 가중합에 ReLU를 적용합니다. 음의 응답을 절댓값으로 바꿔 양의 Fake 근거로 표시하지 않습니다.
- Real로 예측됐더라도 대상은 Fake logit입니다. 자동으로 Real logit 또는 음의 Fake logit으로 바꾸지 않습니다.
- 기존 판정값은 원래의 inference 경로로 계산합니다. 설명용 forward가 이 점수와 일치하는지 검사하고 차이가 허용 범위를 넘으면 설명 저장을 중단합니다.

계산 방식은 [Grad-CAM 원 논문](https://arxiv.org/abs/1610.02391)의 gradient 평균·가중합·ReLU 방식에 따릅니다.

전체 이미지의 종횡비를 유지하는 기존 letterbox 전처리에 맞춰 CAM을 정사각형 입력 좌표로 확대한 뒤 **여백을 제거하고** EXIF 방향이 적용된 원본 크기로 복원합니다. 원본을 crop하거나 입력 크기를 바꾸지 않습니다. 각 이미지의 원본 영역 CAM 최대값으로 나누어 0~1로 정규화합니다. 입력 여백에 있었던 heatmap 값의 비중은 JSON과 한국어 설명에 별도로 기록합니다. 이 비중 역시 heatmap 집계이며 실제 인과적 기여율은 아닙니다.

## 객관적인 설명 범위

원본 좌표를 3×3 격자로 나누고 각 격자의 **양의 heatmap 값 합 / 전체 양의 heatmap 값 합**을 계산합니다. 예를 들면 다음처럼 설명합니다.

> 원본을 3×3 격자로 나눠 집계했을 때 오른쪽 중앙의 heatmap 값 합이 가장 컸습니다(전체 양의 heatmap 값 합의 33.9%). 이 비율은 시각화 값의 공간 분포이며 예측 확률에 대한 기여율이 아닙니다.

메모·상품·얼굴·배경 같은 객체를 탐지하지 않으며 경계가 부자연스럽다거나 질감이 반복된다는 설명을 생성하지 않습니다. uniform map은 특정 위치를 구분하기 어렵다고, zero map은 양의 응답이 없다고 알립니다. 모델의 낮은 확신이나 잘못된 예측도 설명 대상에서 배제하지 않습니다.

**강조된 위치가 실제 AI 생성·조작 영역이라는 증거는 아닙니다.** Grad-CAM은 거친 모델 응답 지도이며 segmentation이나 조작 영역 정답이 아닙니다. 픽셀 수준의 위치 정확도, 인과적 설명, 이미지의 진위 또는 모델 판단의 정당성을 보장하지 않습니다. 서로 다른 이미지의 색 강도를 확률처럼 비교할 수도 없습니다. 이 모델은 앞선 소규모 TIGAS test에서 ROC-AUC 0.50이었으므로 시각화가 생성된다는 사실을 모델 성능의 검증으로 해석하면 안 됩니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_explain -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.check_gradcam --image data/tigas/hub/test/images/DALLE2/1_fake/0220.png --checkpoint outputs/tigas_training/checkpoints/best_model.pth --output-dir outputs/explanations/integration_check
```

`check_gradcam`은 실제 학습 모델의 forward/backward, 기존 예측과의 일치, nonzero finite gradient, 원본 해상도 정렬, 모든 결과 파일, checkpoint 파일 불변을 확인합니다. 단위 테스트는 알려진 Fake logit의 수식, Real 예측에서도 Fake 대상 유지, frozen model, 기존 parameter gradient와 train/eval 상태 보존, 예외 시 hook 제거, 홀수 여백의 Pillow 좌표 일치, 여백만의 응답, 0·균일 map, 3×3 집계, 파일 저장을 검증합니다.

## 이번 구현의 실행 결과

- XAI 테스트 10개, 기존 기능 포함 전체 회귀 테스트 **41개 통과**.
- 실제 TIGAS best checkpoint(epoch 11)와 CUDA에서 384px forward/backward 성공: feature map `(1,1280,12,12)`, finite nonzero gradient 확인.
- CPU에서도 Real 예측 사례로 forward/backward 성공. 예측 확률 0.001036, 판정 Real이지만 설명 대상은 Fake logit임을 확인.
- 일반 예측과 `--explain` 예측의 점수·판정·임계값 일치, checkpoint SHA256 불변 확인.
- 기존 20장 test 평가를 다른 출력 폴더에서 재실행하여 전체 지표와 이미지별 판정이 기존 결과와 일치함을 확인. 기존 학습·평가 코드는 변경하지 않았으며 재학습하지 않음.
- 생성한 그림을 시각 검토했고 원본·heatmap·overlay 해상도 일치, 한국어 TXT/JSON, 모든 파일 저장을 검사함.

예제 그림은 `outputs/explanations/tigas_example/explanation.png`, CPU/GPU 통합 검증은 `outputs/explanations/integration_cpu_real/verification.json`, `integration_cuda/verification.json`, 종합 기록은 `outputs/explanations/xai_verification.json`에 있습니다. Linux에서는 이번 실행을 검증하지 않았습니다.

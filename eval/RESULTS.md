# 스키 사진에서 무릎 굽힘 측정 성능 (EPFL Ski-2DPose)

측정일 2026-11-14. 모든 수치는 `eval/knee_eval.py`, `eval/summarize.py`로 실제 실행한 결과입니다 (CPU 4코어).

## 평가 조건
- 데이터: Ski-2DPose 사진 1,982장 중 양쪽 다리의 골반·무릎·발목이 모두 "보임"으로 표시된 **1,527장**.
- 크기 그룹: 몸통 길이(목~골반 중앙) ÷ 화면 높이. 전체 1,982장의 중앙값 **10.0%**를 기준으로 작은 그룹 727장, 큰 그룹 800장.
  (이전 ChatGPT 작업의 9.25% 기준과 정의가 조금 달라 숫자가 다릅니다.)
- 무릎 각도: 골반-무릎-발목이 이루는 2D 각도(180° = 다리 폄). 정답 좌표로 계산한 각도와의 절대 오차.
- "측정 성공": 양쪽 다리의 골반·무릎·발목 신뢰도가 모두 0.3 이상.
- 좌우: 모델이 뒷모습에서 좌우를 바꾸는 경우가 있어 오차가 작은 쪽으로 매칭 (엄격 매칭 오차는 summary.csv의 `mae_strict_deg`, 약 1° 더 큼).

## 방법
| 방법 | 설명 |
|---|---|
| MoveNet 원본 | MoveNet Thunder v4에 사진 전체를 256×256으로 축소해 입력 |
| MoveNet + 검출 자르기 | YOLO11m 사람 검출기로 가장 확실한 사람 상자 → 정사각형으로 잘라 MoveNet 입력 (실제 앱에서 쓸 수 있는 방식) |
| MoveNet + 정답 자르기 | 정답 좌표로 자른 상자 (참고용 상한선, 앱에서는 불가) |
| YOLO11m-Pose | 사진 전체(1280px)에서 가장 확실한 사람 |

## 결과 (1,527장)
| 방법 | 양쪽 무릎 측정률 (작음 / 큼) | 각도 평균 오차 | 중앙값 | 10° 이내 | 장당 시간 |
|---|---|---|---|---|---|
| MoveNet 원본 | 31.6% (29.4 / 33.5) | 24.3° | 14.2° | 39% | 0.04초 |
| MoveNet + 검출 자르기 | 72.6% (67.7 / 77.1) | 12.7° | 6.5° | 65% | 0.03초 + 검출 0.54초 |
| MoveNet + 정답 자르기 | 62.9% (59.0 / 66.4) | 11.8° | 6.1° | 67% | 0.03초 |
| YOLO11m-Pose | 94.5% (89.5 / 99.0) | 16.5° | 9.1° | 53% | 0.56초 |

같은 사진끼리 비교 (네 방법이 모두 측정한 340장): MoveNet+검출 자르기 8.7°, YOLO-Pose 11.6°, MoveNet 원본 19.5°.
MoveNet+검출 자르기와 YOLO-Pose가 둘 다 측정한 1,082장: 각각 12.7°, 14.7°. YOLO-Pose만 측정한 361장에서 YOLO-Pose 오차는 21.9°.

## 해석
1. **MoveNet에 사진 전체를 넣으면 쓸 수 없습니다.** 측정률 32%, 오차 24°. 스키어가 작은 사진에서 특히 나쁩니다 (30°).
2. **사람 영역을 잘라 넣으면 MoveNet이 가장 정확합니다.** 오차가 절반으로 줄고(12.7°), 작은 그룹과 큰 그룹의 차이도 거의 사라집니다.
3. **YOLO-Pose는 거의 모든 사진에서 값을 내지만 각도는 덜 정확합니다.** 늘어난 측정분(361장)은 오차 22°로 어려운 사진입니다.
4. 정답 자르기가 검출 자르기보다 측정률이 낮은 것은 정답 상자가 스키·폴 끝까지 포함해 사람이 작아지기 때문으로 보입니다.

## 한계
- 정지 사진 기준입니다. 영상에서는 앞뒤 프레임으로 누락을 보간하고 흔들림을 줄일 수 있어 측정률은 더 올라갈 수 있습니다.
- 2D 각도라 정면·후면 촬영에서는 실제 3D 무릎 굽힘과 다릅니다.
- 신뢰도 기준 0.3은 두 모델에 같은 값을 썼지만 모델마다 점수 분포가 달라 측정률 비교는 참고용입니다.
- 사진 속 다른 사람(관중, 코치)이 가장 확실한 사람으로 잡히면 오답이 됩니다 (`knee_hit_rate` 참고).

## 재현
```bash
pip install tensorflow ultralytics opencv-python-headless pandas
# 1단계 (PyTorch): YOLO-Pose + 사람 상자
python eval/knee_eval.py --images Images_webp --labels ski2dpose_labels.json --methods yolo_pose,yolo_det --tag yolo --out results
# 2단계 (TensorFlow): MoveNet 세 가지
python eval/knee_eval.py --images Images_webp --labels ski2dpose_labels.json --movenet movenet_thunder_v4 \
  --methods movenet_full,movenet_detcrop,movenet_gtcrop --boxes results/per_image_yolo.csv --tag movenet --out results
python eval/summarize.py results/per_image_yolo.csv results/per_image_movenet.csv --out results/summary.csv
```
TensorFlow와 PyTorch를 한 프로세스에서 불러오면 LLVM 충돌로 종료되어 두 단계로 나눴습니다.

# 스키 자세 코칭 프로젝트 인계 · 2026-11-16 (오후 갱신)

## 새 세션의 첫 작업

AGENTS.md, docs/prd.md, docs/tech-spec.md, TASKS.md, docs/test-plan.md와 현재 폴더를 먼저 확인한다. 작업 폴더는 D:\ski_imgps이고 GitHub 저장소 yewon51019/ski-computer-vision의 체크아웃이다(브랜치 claude/project-thread-x2c58m). **지금까지의 작업은 커밋하지 않았다.** 커밋과 푸시는 사용자가 요청할 때만 한다.

사용자 요청 사항: 모든 작업 전에 무엇을 할지 설명하고, 임의로 정한 것은 보고한다.

이어서 할 일(순서대로)

1. 웹 화면의 남은 확인: 서버 두 개를 띄우고(아래 "실행") /analysis에서 손촬영 영상으로 다리 선이 스키어 다리 위에 겹치는지 눈으로 확인한다. 타임라인 클릭 이동, 결과 파일 저장, 새 영상, 다른 사람 선택, 후보 0명 안내, 휴대폰 화면도 확인한다. 결과는 docs/test-plan.md의 "웹 화면" 절에 적는다.
   - 2026-11-16 밤 진행분: 일부만 확인했다. 확인된 것과 남은 것은 docs/test-plan.md "웹 화면 · 남은 확인" 절에 있다. 다리 선 겹침 눈 확인은 아직 못 했다.
   - 사용자 새 요청(기록만, 구현 전): 다리만이 아니라 데이터셋의 관절 전부를 다룬다. TASKS.md의 DEV-07 참고. 범위는 EPFL 24개 관절 전부로 정해졌다(사용자, 2026-11-16). MoveNet은 17개만 내므로 모델 선택과 완료 기준을 먼저 정한다.
2. 사용자에게 커밋 여부를 묻는다. 커밋 전 handoff/ 폴더(다른 세션이 만든 중복 사본)는 포함하지 않는다.
3. 사용자와 정할 것: 서버가 받은 영상을 언제 지우는지, 올리기 크기·길이 한도, 지난 분석을 기기에 쌓아 두는 방식, 공개 전 로그인·HTTPS, 사람 검출기 교체(YOLO는 AGPL-3.0).
4. 그 뒤가 DEV-04(턴과 리듬)다. 손촬영 영상은 주행 한 번당 분석 구간이 2초 안팎이라 더 가깝게 또는 원본 해상도로 찍은 영상이 먼저 필요하다.

## 사용자가 정한 것

| 결정 | 내용 |
|---|---|
| 기준 | 사진이 아니라 영상 기준으로 평가한다 |
| GPU | 영상 단계부터 GPU를 쓴다. 사진 평가는 CPU 결과를 채택했다 |
| 보류 | 스키어가 작게 보이는 구간은 보류한다. 기준은 사람 검출 상자 높이 80px |
| 후처리 | 누락 메우기, 흔들림 완화, 카메라 움직임 보정을 넣는다 |
| 사람 검출기 | 추가한다(스키어를 처음 찾는 용도) |
| 스키어 선택 | 사용자가 화면에서 클릭해 고른다(자동 선택 아님) |
| 웹 | FastAPI 서버 + 웹 화면. 처음엔 혼자 쓰지만 다른 사람도 접속할 구조 |
| 보관 | 웹 DB(파이어베이스)를 쓰지 않고 쓰는 기기에 둔다 |
| 결과 표시 | 원본 영상 위에 브라우저가 선을 그린다 |

80px 외의 값(나가는 기준 65px, 최소 지속 0.5초, 가장자리 1%, 메우기·완화 0.2초, 올리기 300MB)은 구현하면서 정한 임시값이다.

## 지금까지의 증거

수치의 근거와 검증 내용은 모두 docs/test-plan.md에 있다.

| 항목 | 결과 |
|---|---|
| 사진(EPFL 1,982장, 화면 전체 입력) | 양쪽 무릎 검출 28.7%, 각도 MAE 왼쪽 25.4°·오른쪽 25.6°. 굽힘을 평균 17.9° 작게 읽는다 |
| 영상(EPFL 158개 구간, 정답 프레임 1,982장) | 화면 전체 29.3%, 추적 59.2%(MAE 13.7°·14.6°), 누락 메우기 후 78.2%. 흔들림 완화로 프레임 간 변화 7.2°→3.3° |
| 카메라 보정 | EPFL에서 검출률을 3.5%p 낮췄다. 손촬영 2편에서는 차이가 없거나 방향이 엇갈려 판단 불가 |
| 사람 검출기 | EPFL 60.2%(메우기 후 79.9%)로 효과 작음. 손촬영에서는 스키어를 잡는 프레임이 28.8%→88.8%로 늘지만 무릎 검출은 20.8% |
| 80px 분석 구간 | EPFL 구간 안 61.6%(메우기 후 81.0%), 오차 중앙값 6.9°. 손촬영 구간 안 81.4%와 90.2%(메우기 후 97.3%, 100%) |
| 손촬영 영상의 한계 | 61초 영상에서 4.7초, 20초 영상에서 1.7초만 분석 구간이다. 720p 카카오톡 전송본이며 원본은 없다. 정답이 없어 각도 오차는 모른다 |
| 웹 | 올리기 → 사람 후보 → 클릭 선택 → 결과 화면까지 브라우저에서 동작 확인. 남은 확인은 위 "이어서 할 일" 1번 |

## 실행

환경은 넷이고 모두 Git에서 제외했다. 없으면 다시 만들어야 한다.

| 환경 | 용도 | 구성 |
|---|---|---|
| .movenet-gpu | MoveNet(GPU) | conda 환경, Python 3.10, TensorFlow 2.10.1, CUDA 11.2, cuDNN 8.1, OpenCV 4.11, numpy<2 |
| .yolo-venv | 사람 검출·추적 | Python 3.12, PyTorch 2.7.1+cu118, ultralytics, lap |
| .web-venv | FastAPI 서버 | Python 3.12, fastapi, uvicorn, python-multipart |
| .movenet-venv | 사진 노트북(CPU) | Python 3.12, TensorFlow 2.21 |

- .movenet-gpu는 D:\anaconda의 conda로는 만들 수 없었다(자체 오류). C:\ProgramData\anaconda3\Scripts\conda.exe로 `-c conda-forge --override-channels python=3.10 pip cudatoolkit=11.2 cudnn=8.1.0`을 설치한 뒤 pip으로 `tensorflow==2.10.1 "numpy<2"`를 넣었다. 스크립트들은 시작할 때 환경의 Library\bin을 PATH에 넣으므로 conda activate 없이 `.movenet-gpu\python.exe 스크립트.py`로 실행된다.
- 모델과 데이터(Git 제외): movenet-tensorflow2-singlepose-thunder-v4/(Kaggle SavedModel), data/models/yolo11m.pt, data/ski2dpose_videos/(EPFL 영상), data/user_videos/(손촬영 2편), ski2dpose_*.zip.

웹 실행(터미널 두 개, 프로젝트 루트에서)

```
.web-venv\Scripts\python.exe -m uvicorn server.app:app --port 8000
npm run dev
```

브라우저에서 http://localhost:5173/analysis 를 연다. 검증 명령은 npm test, npm run lint, npm run build다.

평가 재실행(프로젝트 루트에서)

```
.movenet-gpu\python.exe movenet_video_eval.py --videos data/ski2dpose_videos/Videos
.movenet-gpu\python.exe movenet_video_post.py
.yolo-venv\Scripts\python.exe person_boxes.py data/ski2dpose_videos/Videos --out data/epfl_boxes.csv --weights data/models/yolo11m.pt
.movenet-gpu\python.exe analysis_segments.py --boxes data/epfl_boxes.csv --allframes data/movenet_video_det_allframes.csv --labelled movenet_video_det_frames.csv --out movenet_video_det_segments
```

## 파일 사용법

- 웹: server/app.py(FastAPI), ski_pipeline.py(두 단계 실행), person_tracks.py(사람 찾기·ByteTrack·후보 그림), ski_analyze.py(고른 한 명 분석, result.json), src/analysis.jsx·src/overlay.js(화면), vite.config.js(/api 연결). 작업 파일은 data/jobs/<번호>/에 생긴다.
- 영상 평가: movenet_video_eval.py(화면 전체·추적·카메라 보정·검출 시작 방식 채점), movenet_video_post.py(누락 메우기·흔들림 완화), analysis_segments.py(80px 분석 구간), movenet_video_check.py(정답 없는 영상용, 확인용 영상 생성), person_boxes.py(프레임당 가장 확실한 사람 상자).
- 결과 CSV: movenet_video_*.csv(추적·카메라 보정), movenet_video_det_*.csv(검출 시작·분석 구간). 프레임별 큰 파일은 data/에 있다.
- 사진 평가: movenet_eval.ipynb, movenet_frames.csv, movenet_summary.csv. ski_angles.csv는 사람이 표시한 좌표의 **내각**이며 굽힘량과 비교하려면 180°에서 뺀다.
- eval/은 2026-11-14 Claude 작업(자르기 방식과 YOLO-Pose 비교)이다. check_ski_video.py, run_ski_analysis.cmd, analyze_ski.py는 MediaPipe 시절 도구로 보존만 한다.
- handoff/는 다른 세션이 만든 중복 사본이다. 결과가 루트와 같아 지워도 되지만 사용자 확인 없이 지우지 않는다.
- .claude/launch.json은 미리보기용 서버 설정(ski-api, ski-web)이다.

## 알아 둘 문제

- 사람이 멀리 있을 때는 추적 번호가 자주 바뀐다. 80px 이상 구간에서는 유지돼 후보 목록에는 영향이 없었다.
- 같은 스키어가 여러 번 내려오면 후보가 주행마다 따로 나온다.
- 서버를 내린 뒤에도 uvicorn 프로세스가 8000번 포트에 남은 적이 있다. 서버를 띄우기 전에 포트를 확인한다.
- 분석 도중 서버가 꺼지면 그 작업은 "다시 올려 주세요" 오류가 된다.
- data/jobs/에 시험 때 만든 작업 폴더가 남아 있다.

## 원본 확보

- [EPFL 데이터·논문 및 인용 안내](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/)
- [WebP ZIP](https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_images_webp.zip)
- [정답 좌표 ZIP](https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_labels.json.zip)
- [영상 MP4 ZIP](https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_videos.zip)
- [Google MoveNet Thunder v4](https://www.kaggle.com/models/google/movenet/tensorFlow2/singlepose-thunder/4)

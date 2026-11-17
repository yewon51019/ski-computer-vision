"""웹 분석 2단계(24점 모델): 사용자가 고른 스키어 한 명의 24점 관절과 무릎 굽힘을 구한다.

ski_analyze.py(MoveNet 17점)를 대신하는 2단계다. 2026-11-18 사용자가 24점 모델 채택을 결정했다
(docs/test-plan.md "DEV-07 2단계"). 사람 찾기(1단계)와 같은 PyTorch 환경(.yolo-venv)에서 돈다.

하는 일
  1. 1단계(person_tracks.py)의 tracks.csv에서 고른 사람의 상자를 읽는다.
  2. 프레임마다 스키어 주변을 정사각형으로 잘라 학습한 YOLO11m-Pose(24점)에 넣는다.
     - 고른 사람의 상자가 있는 프레임: 상자 × 1.5 (평가 'detected'와 같은 자르기)
     - 상자가 없는 프레임: 직전 프레임에서 찾은 점들을 감싸는 상자 × 1.3 (학습 자르기와 같음)으로 따라간다.
       찾은 점이 너무 적으면 따라가기를 멈춘다.
  3. 분석 구간 규칙(80px), 누락 메우기, 흔들림 완화는 ski_analyze.py와 같은 함수·같은 기준값을 쓴다.

result.json 형식은 ski_analyze.py와 같고 다음만 다르다.
  skeleton  "ski24". 화면(src/overlay.js)이 이 값을 보고 24점 선 표를 고른다 (없으면 MoveNet 17점으로 본다)
  frames.body / seen   EPFL 순서 24점 (ski24_dataset.POINTS):
              0 머리, 1 목, 2~5 오른쪽 어깨·팔꿈치·손·폴 끝, 6~9 왼쪽 어깨·팔꿈치·손·폴 끝,
              10~12 오른쪽 골반·무릎·발목, 13~15 왼쪽 골반·무릎·발목,
              16~19 오른쪽 스키 앞끝·발끝·뒤꿈치·스키 뒤끝, 20~23 왼쪽 스키 앞끝·발끝·뒤꿈치·스키 뒤끝
  rule.score  0.5 (24점 모델의 '찾음' 기준. 평가 때와 같음)

웹에서의 쓰임 (2026-11-18 사용자 결정: 섞어 쓰기)
  손촬영 영상에서 몸·무릎은 MoveNet이 더 잘 맞아서(docs/test-plan.md) 웹은 두 모델을 섞어 쓴다.
  두 모델이 서로 다른 Python 환경에 있어 한 프로그램에서 돌릴 수 없으므로, ski_pipeline.analyse가
  ① 이 스크립트를 --out ski24.json으로 먼저 돌리고 ② ski_analyze.py(MoveNet)가 --skis ski24.json으로
  스키·발 8점(16~23번)만 가져가 result.json에 붙인다. 두 스크립트는 같은 영상을 처음부터 읽으므로
  n번째 프레임끼리 짝이 맞고, 같은 tracks.csv 상자(같은 추적 번호)를 쓰므로 같은 사람이다.
  --out을 주지 않으면 이 스크립트만으로 만든 24점 result.json을 쓴다(비교·확인용).

실행 (프로젝트 루트에서)
  .yolo-venv\\Scripts\\python.exe ski24_analyze.py clip.mp4 --job-dir data/jobs/clip --track 3
  .yolo-venv\\Scripts\\python.exe ski24_analyze.py clip.mp4 --job-dir data/jobs/clip --track 3 --out ski24.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from analysis_segments import segment_mask
from movenet_video_post import fill_short_gaps, smooth
from ski24_dataset import MARGIN as POINTS_MARGIN, cut, square_crop

WEIGHTS = Path('data/ski24/runs/yolo11m_pose_512/weights/best.pt')   # ski24_train.py가 만든 가장 좋은 가중치
BOX_MARGIN = 1.5          # 사람 검출 상자로 자를 때 여백 (ski24_eval.py의 DETECTED_MARGIN과 같음)
MIN_POINTS = 6            # 상자 없이 따라갈 때, 직전 프레임에서 이만큼은 찾았어야 그 점들로 자른다 (임시값)
LEGS = {'left': (13, 14, 15), 'right': (10, 11, 12)}    # 무릎 굽힘에 쓰는 세 점: 골반, 무릎, 발목

# 분석 기준값. ski_analyze.py의 RULE과 같고 score(관절 '찾음' 기준)와 model만 다르다.
# drift(MoveNet이 다른 사람으로 옮겨 갔는지 판단)는 이 모델에서는 매 프레임 고른 상자로 자르므로 쓰지 않는다.
RULE = dict(
    min_px=80,          # 분석 구간에 들어가는 상자 높이(px). 사용자 결정
    exit_px=65,         # 구간에서 나가는 높이 (임시값)
    min_seconds=0.5,    # 이보다 짧은 구간은 버린다 (임시값)
    edge=0.01,          # 상자가 화면 가장자리 1% 안까지 닿으면 잘린 것으로 보고 분석하지 않는다
    bridge=0.2,         # 구간 사이 끊김이 0.2초 이하면 하나로 잇는다
    max_gap=0.2,        # 무릎 값이 0.2초 이하로 비면 앞뒤 값으로 메운다
    window=0.2,         # 흔들림 완화 창 길이(초)
    score=0.5,          # 관절 점수 기준 (24점 모델 평가와 같음)
    model='ski24 YOLO11m-Pose 512px')


def flexion(hip, knee, ankle):
    """무릎 굽힘(도) = 180 − 무릎에서의 내각. movenet_video_eval.flexion과 같은 정의. 다리 마디 길이가 0이면 None."""
    a = (hip[0] - knee[0], hip[1] - knee[1])        # 무릎 → 골반
    b = (ankle[0] - knee[0], ankle[1] - knee[1])    # 무릎 → 발목
    if not math.hypot(*a) or not math.hypot(*b):
        return None
    return 180 - math.degrees(math.atan2(abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1]))


def read_track(path, track):
    """tracks.csv에서 고른 사람의 상자만 {프레임 번호: (x0, y0, x1, y1) 비율}로 읽는다."""
    with open(path) as f:
        return {int(r['frame']): tuple(float(r[k]) for k in ('x0', 'y0', 'x1', 'y1'))
                for r in csv.DictReader(f) if int(r['track']) == track}


class Ski24:
    """학습한 24점 YOLO11m-Pose. points()는 (잘라 낸 그림 기준 24×2 좌표, 24개 점수)를 돌려준다.

    ski24_eval.py의 Ski24와 같은 설정(512px, conf 0.1, 가장 확실한 한 명)이라 평가 수치가 그대로 적용된다.
    """

    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def points(self, picture):
        result = self.model.predict(picture, imgsz=512, conf=0.1, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return np.zeros((24, 2)), np.zeros(24)          # 아무도 못 찾음: 모든 점 점수 0
        best = int(result.boxes.conf.argmax())
        return result.keypoints.xy[best].cpu().numpy(), result.keypoints.conf[best].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video', type=Path)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--track', type=int, required=True)      # 사용자가 고른 사람의 추적 번호
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    parser.add_argument('--out', default='result.json')         # 작업 폴더 안에 쓸 결과 파일 이름
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    mine = read_track(args.job_dir / 'tracks.csv', args.track)
    assert mine, 'track %d is not in tracks.csv' % args.track
    model = Ski24(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = None             # 직전 프레임에서 찾은 점들(픽셀, 점수 기준 이상인 것만). 상자 없는 프레임에서 따라가는 데 쓴다
    raw = {'left': [], 'right': []}          # 프레임별 무릎 굽힘 원래 값 (못 찾으면 NaN)
    body, seen, boxes, box_px, inside = [], [], [], [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        box = mine.get(i)
        # 어디를 자를지: 고른 사람의 상자가 있으면 그 상자, 없으면 직전에 찾은 점들, 둘 다 없으면 이 프레임은 건너뛴다
        if box is not None:
            crop = square_crop(np.array([[box[0] * w, box[1] * h], [box[2] * w, box[3] * h]]), BOX_MARGIN)
        elif last is not None:
            crop = square_crop(last, POINTS_MARGIN)
        else:
            crop = None
        if crop is None:
            xy, score = None, None
        else:
            picture, (x0, y0, _, _) = cut(bgr, crop)
            xy, score = model.points(picture)
            xy = xy + [x0, y0]                               # 잘라 낸 그림 좌표 + 자른 위치 = 원본 프레임 좌표(픽셀)
            good = score >= RULE['score']
            last = xy[good] if good.sum() >= MIN_POINTS else None   # 너무 적게 찾았으면 따라가기를 멈춘다

        for side, ids in LEGS.items():
            bend = flexion(*xy[list(ids)]) if xy is not None and np.all(score[list(ids)] >= RULE['score']) else None
            raw[side].append(np.nan if bend is None else bend)
        # 화면에 그릴 관절: 무릎 각도와 달리 메우기·완화를 하지 않은 그 프레임의 원래 값 (원본 대비 비율)
        body.append(None if xy is None else (xy / [w, h]).round(4).tolist())
        seen.append(None if xy is None else (score >= RULE['score']).astype(int).tolist())
        boxes.append(None if box is None else [round(v, 4) for v in box])
        box_px.append(np.nan if box is None else (box[3] - box[1]) * h)     # 상자 높이(px), 분석 구간 판정용
        e = RULE['edge']
        inside.append(box is not None and box[0] > e and box[1] > e and box[2] < 1 - e and box[3] < 1 - e)
        i += 1
        if args.progress and i % 20 == 0:
            # part='skis': 같은 '관절 분석' 단계 안에서 스키를 찾는 앞부분임을 화면에 알린다
            (args.job_dir / 'progress.json').write_text(
                json.dumps(dict(stage='pose', part='skis', done=i, total=total)))
    cap.release()

    # 여기부터는 ski_analyze.py와 같은 후처리: 분석 구간 → 짧은 빈칸 메우기 → 흔들림 완화 → 구간 밖 보류
    on = segment_mask(np.array(box_px, float), np.array(inside, bool), fps, RULE['min_px'], RULE['exit_px'],
                      RULE['min_seconds'], RULE['bridge'])
    max_gap, window = max(1, round(RULE['max_gap'] * fps)), max(3, round(RULE['window'] * fps) | 1)
    out = {}
    for side, values in raw.items():
        values = np.array(values, float)
        shown = smooth(fill_short_gaps(values, max_gap), window)
        shown[~on] = np.nan
        out[side] = [None if np.isnan(v) else round(float(v), 1) for v in shown]
        out['found_' + side] = (~np.isnan(values)).astype(int).tolist()
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    segments = [dict(start=round(a / fps, 3), end=round(b / fps, 3)) for a, b in zip(edges[::2], edges[1::2])]
    both = sum(l is not None and r is not None for l, r in zip(out['left'], out['right']))
    result = dict(
        skeleton='ski24', video=dict(width=w, height=h, fps=fps, frames=i), track=args.track, rule=RULE,
        segments=segments,
        summary=dict(seconds=round(i / fps, 2), analysed_seconds=round(float(on.sum()) / fps, 2),
                     both_knees_pct_in_segments=round(100 * both / max(1, int(on.sum())), 1)),
        frames=dict(seg=on.astype(int).tolist(), left=out['left'], right=out['right'], body=body, seen=seen,
                    found=[list(p) for p in zip(out['found_left'], out['found_right'])], box=boxes))
    (args.job_dir / args.out).write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    if args.progress and args.out == 'result.json':     # 섞어 쓸 때는 MoveNet 단계가 이어지므로 '끝'을 쓰지 않는다
        (args.job_dir / 'progress.json').write_text(json.dumps(dict(stage='done', done=i, total=i)))
    print('frames %d | segments %d | analysed %.1fs of %.1fs | both knees in segments %.1f%%'
          % (i, len(segments), on.sum() / fps, i / fps, result['summary']['both_knees_pct_in_segments']))
    for s in segments:
        print('  %.2f - %.2f s' % (s['start'], s['end']))


if __name__ == '__main__':
    main()

"""웹 분석 2단계: 사용자가 고른 스키어 한 명의 관절과 무릎 굽힘을 구한다.

TensorFlow 환경(.movenet-gpu)에서 돈다. 1단계(person_tracks.py)의 tracks.csv에서 고른 사람의 상자를 읽고,
그 사람을 MoveNet으로 따라가며 관절을 찾는다. 그다음 분석 구간 규칙(80px), 누락 메우기, 흔들림 완화를
적용해 브라우저가 원본 영상 위에 그릴 result.json을 만든다.

result.json
  video     원본 너비, 높이, fps, 프레임 수
  rule      이번 분석에 쓴 기준값들 (아래 RULE)
  segments  [{start, end}] 초 단위. 스키어가 충분히 크게 보여 분석한 구간
  frames    프레임마다 하나씩 값이 들어 있는 배열들:
              seg    분석 구간 안이면 1, 밖이면 0
              left / right   왼쪽·오른쪽 무릎 굽힘(도, 흔들림 완화 후). 보류면 null
              body   MoveNet 17점(COCO 순서: 코, 눈, 귀, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목.
                     각각 왼쪽 먼저)의 [x, y]. 원본 프레임 대비 비율(0~1)이라 화면 크기와 상관없이 그릴 수 있다.
                     스키어를 아직 찾기 전 프레임은 null
              seen   17점 각각, 이 프레임에서 모델 점수가 rule.score(0.30) 이상이면 1. 화면은 0인 점에 닿는 선을 흐리게 그린다
              found  [왼쪽, 오른쪽] 이 프레임에서 모델이 그 다리(골반·무릎·발목)를 직접 찾았으면 1
                     (0이면 left/right 값은 앞뒤 프레임으로 메운 값이거나 null)
              box    고른 스키어의 상자 [x0, y0, x1, y1] 또는 null
              skis   (--skis를 줬을 때만) 24점 모델(ski24_analyze.py)이 찾은 스키·발 8점의 [x, y] 비율.
                     순서: 오른쪽 스키 앞끝·발끝·뒤꿈치·스키 뒤끝, 왼쪽 스키 앞끝·발끝·뒤꿈치·스키 뒤끝
                     (EPFL 16~23번). 그 모델이 스키어를 따라가지 못한 프레임은 null
              skis_seen  8점 각각 24점 모델 점수가 0.5 이상이면 1
  skis_rule (--skis를 줬을 때만) 스키 점을 만든 24점 모델의 기준값

섞어 쓰기 (2026-11-18 사용자 결정)
  몸 17점과 무릎 수치는 MoveNet, 스키·발 점은 24점 모델에서 가져온다. 폴 끝은 24점 모델도 거의 못 맞혀서
  (PCK 5%) 가져오지 않는다. 두 결과는 같은 영상을 처음부터 읽은 프레임 번호로 짝짓고, 프레임 수가 다르면 멈춘다.

실행
  .movenet-gpu\\python.exe ski_analyze.py clip.mp4 --job-dir data/jobs/clip --track 3
  .movenet-gpu\\python.exe ski_analyze.py clip.mp4 --job-dir data/jobs/clip --track 3 --skis ski24.json
"""
import argparse
import json
from pathlib import Path

import movenet_video_eval as ev            # CUDA 경로 설정, TensorFlow, MoveNet, 다음 프레임 자르기 규칙 (TensorFlow보다 먼저)
from analysis_segments import segment_mask
from movenet_video_post import fill_short_gaps, smooth

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# 분석 기준값. 결과 파일에도 그대로 적어 어떤 기준으로 만든 결과인지 남긴다.
RULE = dict(
    min_px=80,          # 분석 구간에 들어가는 상자 높이(px). 사용자 결정
    exit_px=65,         # 구간에서 나가는 높이. 들어가는 기준보다 낮게 해 경계에서 켜졌다 꺼졌다 하지 않게 (임시값)
    min_seconds=0.5,    # 이보다 짧은 구간은 버린다 (임시값)
    edge=0.01,          # 상자가 화면 가장자리 1% 안까지 닿으면 잘린 것으로 보고 분석하지 않는다
    bridge=0.2,         # 구간 사이 끊김이 0.2초 이하면 하나로 잇는다
    max_gap=0.2,        # 무릎 값이 0.2초 이하로 비면 앞뒤 값으로 메운다
    window=0.2,         # 흔들림 완화 창 길이(초)
    score=ev.KP_THRESH,  # 관절 점수 기준 0.30 (사진 평가와 같음)
    drift=0.25)         # MoveNet 골반이 고른 사람 상자에서 상자 크기의 25% 넘게 벗어나면 다른 사람을 본 것으로 판단


SKI_POINTS = slice(16, 24)    # 24점(EPFL 순서) 중 스키·발 8점: 오른쪽 16~19, 왼쪽 20~23


def add_skis(result, ski24):
    """24점 모델 결과(ski24)에서 스키·발 8점만 꺼내 MoveNet 결과(result)의 frames.skis / skis_seen에 붙인다.

    두 결과는 같은 영상을 처음부터 끝까지 읽어 만든 것이라 프레임 번호가 곧 짝이다.
    프레임 수가 다르면(다른 영상이거나 읽다가 끊김) 잘못 짝지어지므로 붙이지 않고 멈춘다.
    """
    frames = result['video']['frames']
    if ski24.get('skeleton') != 'ski24' or ski24['video']['frames'] != frames:
        raise ValueError('ski24 result does not match this video (%s frames vs %d)' % (ski24['video']['frames'], frames))
    body, seen = ski24['frames']['body'], ski24['frames']['seen']
    result['frames']['skis'] = [None if p is None else p[SKI_POINTS] for p in body]
    result['frames']['skis_seen'] = [None if s is None else s[SKI_POINTS] for s in seen]
    result['skis_rule'] = ski24['rule']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video', type=Path)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--track', type=int, required=True)      # 사용자가 고른 사람의 추적 번호
    parser.add_argument('--model', type=Path, default=Path('movenet-tensorflow2-singlepose-thunder-v4'))
    parser.add_argument('--skis', help='작업 폴더 안의 24점 결과 파일(ski24_analyze.py --out). 주면 스키·발 8점을 붙인다')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    # 고른 사람이 보이는 프레임과 상자만 남긴다 (프레임 번호로 바로 찾을 수 있게 색인)
    tracks = pd.read_csv(args.job_dir / 'tracks.csv')
    mine = tracks[tracks.track == args.track].set_index('frame')
    assert len(mine), 'track %d is not in tracks.csv' % args.track
    for gpu in tf.config.list_physical_devices('GPU'):         # GPU 메모리는 필요한 만큼만
        tf.config.experimental.set_memory_growth(gpu, True)
    net = ev.MoveNet(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    region, tracking, i = None, False, 0     # region: 이번 프레임에서 MoveNet에 넣을 영역, tracking: 스키어를 따라가는 중인지
    raw = {'left': [], 'right': []}          # 프레임별 무릎 굽힘 원래 값 (못 찾으면 NaN)
    body, seen, boxes, box_px, inside = [], [], [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        box = tuple(mine.loc[i, ['x0', 'y0', 'x1', 'y1']]) if i in mine.index else None
        # 고른 사람이 처음 나타나면 그 상자에서 따라가기 시작한다
        if box is not None and not tracking:
            region, tracking = ev.box_region(box, h, w), True
        if region is None:
            region = ev.whole_frame(h, w)
        xy, score = net.crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), region)
        hips = (xy[ev.L_HIP] + xy[ev.R_HIP]) / 2 / [w, h]      # 양쪽 골반 가운데(비율)
        if box is not None:
            # MoveNet이 고른 사람에게서 벗어났다(다른 사람을 봄): 고른 사람의 상자에서 다시 찾는다
            grow_x, grow_y = (box[2] - box[0]) * RULE['drift'], (box[3] - box[1]) * RULE['drift']
            if not (box[0] - grow_x <= hips[0] <= box[2] + grow_x and box[1] - grow_y <= hips[1] <= box[3] + grow_y):
                region = ev.box_region(box, h, w)
                xy, score = net.crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), region)
        found = ev.legs(xy, score)                              # 다리별 굽힘, 점수가 모자라면 None
        for side in raw:
            raw[side].append(np.nan if found[side] is None else found[side])
        # 화면에 그릴 관절 저장. 무릎 각도와 달리 누락 메우기·흔들림 완화를 하지 않은 그 프레임의 원래 값이다
        skier_known = tracking or box is not None          # 추적 중이거나 이 프레임에 고른 사람의 상자가 있음
        body.append((xy / [w, h]).round(4).tolist() if skier_known else None)
        seen.append((score >= RULE['score']).astype(int).tolist() if skier_known else None)
        boxes.append(None if box is None else [round(float(v), 4) for v in box])
        box_px.append(np.nan if box is None else (box[3] - box[1]) * h)     # 상자 높이(px), 분석 구간 판정용
        e = RULE['edge']
        inside.append(box is not None and box[0] > e and box[1] > e and box[2] < 1 - e and box[3] < 1 - e)
        region, tracking = ev.next_region(xy, score, h, w)     # 이번에 찾은 관절로 다음 프레임 영역을 정한다
        i += 1
        if args.progress and i % 20 == 0:
            (args.job_dir / 'progress.json').write_text(json.dumps(dict(stage='pose', done=i, total=total)))
    cap.release()

    # 분석 구간: 상자 높이와 가장자리 조건으로 프레임마다 켜짐/꺼짐
    on = segment_mask(np.array(box_px, float), np.array(inside, bool), fps, RULE['min_px'], RULE['exit_px'],
                      RULE['min_seconds'], RULE['bridge'])
    # 초 단위 기준을 프레임 수로. 완화 창은 홀수여야 가운데가 있으므로 | 1로 홀수로 만든다(최소 3)
    max_gap, window = max(1, round(RULE['max_gap'] * fps)), max(3, round(RULE['window'] * fps) | 1)
    out = {}
    for side, values in raw.items():
        values = np.array(values, float)
        shown = smooth(fill_short_gaps(values, max_gap), window)   # 짧은 빈칸 메우기 → 흔들림 완화
        shown[~on] = np.nan                                     # 분석 구간 밖은 보류
        out[side] = [None if np.isnan(v) else round(float(v), 1) for v in shown]
        out['found_' + side] = (~np.isnan(values)).astype(int).tolist()
    # 켜짐/꺼짐이 바뀌는 프레임을 찾아 [시작, 끝] 구간 목록으로 바꾼다
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    segments = [dict(start=round(a / fps, 3), end=round(b / fps, 3)) for a, b in zip(edges[::2], edges[1::2])]
    both = sum(l is not None and r is not None for l, r in zip(out['left'], out['right']))   # 양쪽 무릎 다 있는 프레임 수
    result = dict(
        video=dict(width=w, height=h, fps=fps, frames=i), track=args.track, rule=RULE, segments=segments,
        summary=dict(seconds=round(i / fps, 2), analysed_seconds=round(float(on.sum()) / fps, 2),
                     both_knees_pct_in_segments=round(100 * both / max(1, int(on.sum())), 1)),
        frames=dict(seg=on.astype(int).tolist(), left=out['left'], right=out['right'], body=body, seen=seen,
                    found=[list(p) for p in zip(out['found_left'], out['found_right'])], box=boxes))
    if args.skis:
        add_skis(result, json.loads((args.job_dir / args.skis).read_text(encoding='utf-8')))
    # 공백 없이 저장해 파일을 작게 (브라우저가 내려받는 크기)
    (args.job_dir / 'result.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    if args.progress:
        (args.job_dir / 'progress.json').write_text(json.dumps(dict(stage='done', done=i, total=i)))
    print('frames %d | segments %d | analysed %.1fs of %.1fs | both knees in segments %.1f%%'
          % (i, len(segments), on.sum() / fps, i / fps, result['summary']['both_knees_pct_in_segments']))
    for s in segments:
        print('  %.2f - %.2f s' % (s['start'], s['end']))


if __name__ == '__main__':
    main()

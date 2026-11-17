"""영상에서 스키어가 충분히 크게 보여 분석할 수 있는 구간을 표시하고, 그 구간 안에서 채점한다.

프레임이 분석 구간에 들어가는 조건
  크기       사람 상자 높이가 --min-px(80px) 이상. 한 번 들어가면 --exit-px(65px) 아래로 떨어질 때까지 유지
             (들어가는 기준과 나가는 기준을 달리해 경계 근처에서 켜졌다 꺼졌다 반복하지 않게)
  화면 안    상자가 화면 가장자리(너비·높이의 --edge 이내)에 닿지 않음 (잘린 사람 제외)
  지속 시간  위 조건이 --min-seconds 이상 이어짐
상자가 --bridge초 이하로 잠깐 사라지면 마지막 상자를 이어서 쓴다.

모델은 돌리지 않는다. 입력은 사람 상자(person_boxes.py)와 프레임별 무릎 굽힘
(movenet_video_eval.py --allframes, 또는 movenet_video_check.py의 <out>_frames.csv)이다.
segment_mask는 웹 분석(ski_analyze.py)에서도 같은 함수를 가져다 쓴다.

실행 (프로젝트 루트에서)
  EPFL:     python analysis_segments.py --boxes data/epfl_boxes.csv --allframes data/movenet_video_det_allframes.csv
                --labelled movenet_video_det_frames.csv --out movenet_video_det_segments
  영상 한 편: python analysis_segments.py --boxes data/clip_boxes.csv --allframes data/clip_frames.csv
                --height 720 --fps 24 --out data/clip_segments
"""
import argparse

import numpy as np
import pandas as pd

from movenet_video_post import fill_short_gaps, smooth


def segment_mask(box_px, inside, fps, min_px, exit_px, min_seconds, bridge):
    """프레임마다 분석 구간 안인지(True/False) 배열을 돌려준다.

    box_px: 프레임별 상자 높이(px, 없으면 NaN), inside: 상자가 화면 가장자리에 닿지 않았는지.
    순서: 1) 짧게 사라진 상자 이어 붙이기 → 2) 크기·화면 안 조건으로 켜짐/꺼짐 → 3) 너무 짧은 구간 버리기
    """
    n = len(box_px)
    px, ok_inside = box_px.copy(), inside.copy()
    # 1) 상자가 bridge초 이하로 비면 마지막 상자 값을 이어 쓴다
    limit, last, age = max(1, round(bridge * fps)), None, 0    # last: 마지막으로 상자가 있던 프레임, age: 그 뒤 지난 프레임 수
    for i in range(n):
        if np.isnan(px[i]):
            age += 1
            if last is not None and age <= limit:
                px[i], ok_inside[i] = px[last], ok_inside[last]
        else:
            last, age = i, 0
    # 2) 직전에 켜져 있었으면 나가는 기준(exit_px), 꺼져 있었으면 들어가는 기준(min_px)을 적용
    on, state = np.zeros(n, bool), False
    for i in range(n):
        good = not np.isnan(px[i]) and ok_inside[i]
        state = good and (px[i] >= (exit_px if state else min_px))
        on[i] = state
    # 3) min_seconds보다 짧게 켜진 구간은 끈다
    need, i = max(1, round(min_seconds * fps)), 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i                                       # i부터 j 직전까지 켜진 구간 하나
        while j < n and on[j]:
            j += 1
        if j - i < need:
            on[i:j] = False
        i = j
    return on


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--boxes', required=True)
    parser.add_argument('--allframes', required=True)
    parser.add_argument('--labelled', help='labelled-frame CSV with ground truth (EPFL)')
    parser.add_argument('--height', type=int, help='frame height in pixels (single video)')
    parser.add_argument('--fps', type=float, help='frames per second (single video)')
    parser.add_argument('--method', default='trackdet')        # 어느 추적 방식의 무릎 값을 쓸지
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-px', type=float, default=80)
    parser.add_argument('--exit-px', type=float, default=65)
    parser.add_argument('--min-seconds', type=float, default=0.5)
    parser.add_argument('--edge', type=float, default=0.01)
    parser.add_argument('--bridge', type=float, default=0.2)
    parser.add_argument('--max-gap', type=float, default=0.2)
    parser.add_argument('--window', type=float, default=0.2)
    args = parser.parse_args()
    m = args.method

    # 프레임별 무릎 값에 사람 상자를 붙인다. 영상 한 편이면 video·split 칸이 비어 있으므로 ''로 맞춘다
    keys = {'video': str, 'split': str}
    boxes = pd.read_csv(args.boxes, dtype=keys).fillna({'video': '', 'split': ''})
    every = pd.read_csv(args.allframes, dtype=keys)
    for column in ('video', 'split'):
        if column not in every:
            every[column] = ''
    every = every.fillna({'video': '', 'split': ''}).merge(
        boxes[['video', 'split', 'frame', 'x0', 'y0', 'x1', 'y1']], on=['video', 'split', 'frame'], how='left')
    labelled = pd.read_csv(args.labelled, dtype=keys) if args.labelled else None
    if labelled is not None:
        # EPFL: 시퀀스마다 화면 높이·fps가 다르므로 정답 파일에서 가져온다
        meta = labelled.groupby(['video', 'split']).agg(height=('height', 'first'), fps=('fps', 'first'))

    parts, segments = [], []
    for (video, split), seq in every.groupby(['video', 'split'], sort=False):
        seq = seq.sort_values('frame').copy()
        height, fps = (meta.loc[(video, split)] if labelled is not None else (args.height, args.fps))
        box_px = ((seq.y1 - seq.y0) * height).to_numpy(float)     # 상자 높이(px)
        e = args.edge
        inside = ((seq.x0 > e) & (seq.y0 > e) & (seq.x1 < 1 - e) & (seq.y1 < 1 - e)).to_numpy(bool)
        seq['box_px'] = box_px
        seq['in_segment'] = segment_mask(box_px, inside, fps, args.min_px, args.exit_px, args.min_seconds, args.bridge)
        # 무릎 값 세 단계(원래 값, 메운 값, 완화한 값)도 만들어 둔다
        max_gap, window = max(1, round(args.max_gap * fps)), max(3, round(args.window * fps) | 1)
        for side in ('left', 'right'):
            raw = seq[f'{m}_{side}'].to_numpy(float)
            filled = fill_short_gaps(raw, max_gap)
            seq[f'raw_{side}'], seq[f'filled_{side}'], seq[f'smoothed_{side}'] = raw, filled, smooth(filled, window)
        # 켜짐/꺼짐이 바뀌는 곳으로 구간 목록 만들기
        on = seq.in_segment.to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        frames = seq.frame.to_numpy()
        for a, b in zip(edges[::2], edges[1::2]):
            segments.append(dict(video=video, split=split, start_frame=int(frames[a]), end_frame=int(frames[b - 1]),
                                 seconds=round((b - a) / fps, 2), box_px_median=float(np.nanmedian(box_px[a:b]))))
        seq['fps'] = fps
        parts.append(seq)
    df = pd.concat(parts)
    seg = pd.DataFrame(segments)
    seg.to_csv(args.out + '.csv', index=False)

    # 구간 안·밖·전체에서 양쪽 무릎을 다 찾은 비율
    rows = []
    for name, g in (('in segment', df[df.in_segment]), ('outside', df[~df.in_segment]), ('all', df)):
        r = dict(part=name, frames=len(g), seconds=round((1 / g.fps).sum(), 1), share_pct=100 * len(g) / len(df))
        for stage in ('raw', 'filled'):
            r[stage + '_both_pct'] = 100 * (g[stage + '_left'].notna() & g[stage + '_right'].notna()).mean() if len(g) else np.nan
        rows.append(r)
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 230)
    print('rule: enter >= %gpx, leave < %gpx, >= %.1fs, edge %.0f%% | method %s' % (args.min_px, args.exit_px, args.min_seconds, 100 * args.edge, m))
    print('segments: %d | median length %.1fs | longest %.1fs' % (len(seg), seg.seconds.median() if len(seg) else 0, seg.seconds.max() if len(seg) else 0))
    print(table.round(1).to_string(index=False))

    if labelled is not None:
        # 정답이 있으면(EPFL) 구간 안·밖에서 각도 오차까지 채점한다
        lab = labelled.merge(df[['video', 'split', 'frame', 'in_segment', 'box_px'] +
                                [f'{s}_{side}' for s in ('raw', 'filled', 'smoothed') for side in ('left', 'right')]],
                             left_on=['video', 'split', 'frame_idx'], right_on=['video', 'split', 'frame'], how='left')
        out = []
        for name, g in (('in segment', lab[lab.in_segment == True]), ('outside', lab[lab.in_segment != True]), ('all', lab)):
            for stage in ('raw', 'filled', 'smoothed'):
                both = g[stage + '_left'].notna() & g[stage + '_right'].notna()
                signed = pd.concat([g[stage + '_left'] - g.gt_left, g[stage + '_right'] - g.gt_right]).dropna()   # +면 더 굽혔다고 읽음
                err = signed.abs()
                out.append(dict(part=name, stage=stage, labelled=len(g), both_pct=100 * both.mean(), legs=len(err),
                                mae_deg=err.mean(), median_deg=err.median(), within10_pct=100 * (err <= 10).mean(),
                                over45_pct=100 * (err > 45).mean(), signed_mean_deg=signed.mean()))
        scored = pd.DataFrame(out)
        scored.to_csv(args.out + '_scores.csv', index=False)
        print('labelled frames against ground truth:')
        print(scored.round(1).to_string(index=False))
    else:
        print(seg.to_string(index=False))


if __name__ == '__main__':
    main()

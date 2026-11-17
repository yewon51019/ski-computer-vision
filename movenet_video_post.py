"""movenet_video_eval.py가 낸 프레임별 무릎 굽힘에 누락 메우기와 흔들림 완화를 적용하고 평가한다.

시퀀스마다, 다리별로 시간에 따른 굽힘 값을 세 단계로 만든다.

  raw       모델이 낸 그대로. 다리를 못 찾은 프레임은 비어 있다
  filled    --max-gap초 이하의 빈칸을 앞뒤 값 사이 직선으로 잇는다.
            빈칸 양쪽 모두에 값이 있을 때만 메우고, 더 긴 빈칸은 그대로 비워 둔다(추측하지 않음)
  smoothed  filled에 --window초 길이의 가운데 정렬 중앙값을 적용한다(튀는 값 완화)

각 단계를 정답 프레임에서 다른 평가와 같은 규칙으로 정답과 비교한다. 여기서는 모델을 돌리지 않는다.
fill_short_gaps·smooth는 웹 분석(ski_analyze.py)에서도 같은 함수를 가져다 쓴다.

실행 (프로젝트 루트에서)
  .movenet-gpu\\python.exe movenet_video_post.py
"""
import argparse

import numpy as np
import pandas as pd

KNOWN_TRACKERS = ('track', 'trackcam', 'trackdet')   # 입력 파일에 있으면 처리할 추적 방식들
STAGES = ('raw', 'filled', 'smoothed')


def fill_short_gaps(values, max_gap):
    """연속된 빈칸(NaN)이 max_gap 프레임 이하이고 양쪽에 값이 있으면 직선으로 메운다.

    영상 맨 앞·맨 뒤의 빈칸(한쪽에 값이 없음)과 긴 빈칸은 그대로 둔다.
    """
    out = values.copy()
    missing = np.isnan(values)
    n, i = len(values), 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i                                   # i부터 j 직전까지가 빈칸 하나
        while j < n and missing[j]:
            j += 1
        if i > 0 and j < n and j - i <= max_gap:
            out[i:j] = np.interp(np.arange(i, j), [i - 1, j], [values[i - 1], values[j]])
        i = j
    return out


def smooth(values, window):
    """가운데 정렬 이동 중앙값. 평균 대신 중앙값이라 한 프레임만 튄 값에 덜 끌려간다.

    원래 비어 있던 프레임은 계속 비워 둔다(완화가 빈칸을 메우지 않게).
    """
    result = pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    result[np.isnan(values)] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--allframes', default='data/movenet_video_allframes.csv')
    parser.add_argument('--frames', default='movenet_video_frames.csv')
    parser.add_argument('--out', default='movenet_video_post')
    parser.add_argument('--max-gap', type=float, default=0.2, help='longest gap to fill, seconds')
    parser.add_argument('--window', type=float, default=0.2, help='smoothing window, seconds')
    args = parser.parse_args()

    every = pd.read_csv(args.allframes, dtype={'video': str, 'split': str})       # 모든 프레임
    labelled = pd.read_csv(args.frames, dtype={'video': str, 'split': str})       # 정답 프레임 (정답 각도 포함)
    fps = labelled.groupby(['video', 'split']).fps.first()
    TRACKERS = [m for m in KNOWN_TRACKERS if m + '_left' in every.columns]

    # 단계별 집계: 프레임 수, 양쪽 다 있는 프레임 수, 프레임 간 변화량 합(흔들림 정도)
    stats = {(m, s): dict(frames=0, both=0, jitter_sum=0.0, jitter_n=0) for m in TRACKERS for s in STAGES}
    parts = []
    for (video, split), seq in every.groupby(['video', 'split'], sort=False):
        seq = seq.sort_values('frame')
        rate = fps[video, split]
        # 초 단위 기준을 이 영상의 프레임 수로 바꾼다
        max_gap = max(1, round(args.max_gap * rate))
        window = max(3, round(args.window * rate) | 1)          # 가운데가 있도록 홀수 프레임 (최소 3)
        out = seq[['video', 'split', 'frame']].copy()
        for method in TRACKERS:
            series = {}
            for side in ('left', 'right'):
                raw = seq[f'{method}_{side}'].to_numpy(float)
                filled = fill_short_gaps(raw, max_gap)
                series[side] = dict(raw=raw, filled=filled, smoothed=smooth(filled, window))
                for stage in STAGES:
                    out[f'{method}_{stage}_{side}'] = series[side][stage]
                out[f'{method}_wasfilled_{side}'] = np.isnan(raw) & ~np.isnan(filled)   # 메워서 생긴 값 표시
            for stage in STAGES:
                st = stats[method, stage]
                left, right = series['left'][stage], series['right'][stage]
                st['frames'] += len(left)
                st['both'] += int((~np.isnan(left) & ~np.isnan(right)).sum())
                for values in (left, right):                     # 프레임 사이 변화량 (흔들림 지표)
                    step = np.abs(np.diff(values))
                    st['jitter_sum'] += float(np.nansum(step))
                    st['jitter_n'] += int((~np.isnan(step)).sum())
        parts.append(out)
    processed = pd.concat(parts)
    # 정답 프레임에 처리 결과를 붙인다 (영상·시퀀스·프레임 번호로 맞춤)
    df = labelled.merge(processed, left_on=['video', 'split', 'frame_idx'], right_on=['video', 'split', 'frame'],
                        how='left')

    rows = []
    for method in TRACKERS:
        for stage in STAGES:
            for name, g in [('all', df), *df.groupby('group')]:
                left, right = g[f'{method}_{stage}_left'], g[f'{method}_{stage}_right']
                both = left.notna() & right.notna()
                gt_both = g.gt_left.notna() & g.gt_right.notna()
                r = dict(method=method, stage=stage, group=name, frames=len(g), both_pred=int(both.sum()),
                         coverage_pct=100 * both.mean(), both_gt=int(gt_both.sum()),
                         both_paired=int((gt_both & both).sum()))
                for side, pred in (('left', left), ('right', right)):
                    errors = (g['gt_' + side] - pred).abs().dropna()
                    r[side + '_mae_deg'] = errors.mean()
                    r[side + '_pairs'] = len(errors)
                    r[side + '_median_deg'] = errors.median()
                if stage != 'raw':                               # 메워 넣은 값만 따로 본 오차 (메우기가 믿을 만한지)
                    e = pd.concat([(g['gt_' + s] - g[f'{method}_{stage}_{s}']).abs()[g[f'{method}_wasfilled_{s}'] == True]
                                   for s in ('left', 'right')]).dropna()
                    r['filled_mae_deg'], r['filled_pairs'] = e.mean(), len(e)
                if name == 'all':
                    st = stats[method, stage]
                    r['allframes_both_pct'] = 100 * st['both'] / st['frames']
                    r['step_deg_per_frame'] = st['jitter_sum'] / max(1, st['jitter_n'])   # 프레임당 평균 변화(도)
                rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out + '_summary.csv', index=False)
    pd.set_option('display.width', 260)
    pd.set_option('display.max_columns', 30)
    print('max gap %.2f s | window %.2f s | frames %d | labelled %d' % (args.max_gap, args.window, len(every), len(df)))
    print(summary[summary.group == 'all'].drop(columns=['group', 'frames', 'both_gt']).round(2).to_string(index=False))
    print()
    print(summary[summary.group.isin(['small', 'large'])][
        ['method', 'stage', 'group', 'frames', 'both_pred', 'coverage_pct', 'left_mae_deg', 'left_pairs',
         'right_mae_deg', 'right_pairs']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()

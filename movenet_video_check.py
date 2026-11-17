"""정답이 없는 영상(손촬영 등)에 MoveNet 추적을 돌려, 무엇이 보이는지 보고하고 확인용 영상을 만든다.

추적 방식은 movenet_video_eval.py(track, trackcam, --boxes가 있으면 trackdet)와 같고,
누락 메우기·흔들림 완화는 movenet_video_post.py와 같다. 정답이 없으므로 각도 오차 대신
검출률, 프레임 간 변화량, 스키어를 놓친 위치를 보고하고, 눈으로 판단할 수 있게 선을 그린 영상을 저장한다.

만드는 것
  <out>_frames.csv     모든 프레임: 방식별 무릎 굽힘, 영역, 카메라 움직임 (data/ 아래)
  <out>_annotated.mp4  추적 영역(흰 상자), 다리 선(빨강, 기준 넘으면 굵게·못 넘으면 가늘게), 왼쪽 위에 시각과 각도
  <out>_still_*.jpg    영상 전체에 고르게 뽑은 확인용 정지 그림 (--stills장)
  <out>_summary.csv    방식 × 단계(원래·메움·완화)별 검출률

실행 (프로젝트 루트에서)
  .movenet-gpu\\python.exe movenet_video_check.py data/user_videos/clip.mp4
"""
import argparse
import math
import time
from pathlib import Path

import movenet_video_eval as ev        # CUDA 경로 설정과 TensorFlow 불러오기 (반드시 TensorFlow보다 먼저)
from movenet_video_post import fill_short_gaps, smooth

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video', type=Path)
    parser.add_argument('--model', type=Path, default=Path('movenet-tensorflow2-singlepose-thunder-v4'))
    parser.add_argument('--out', help='output prefix (default: data/<video name>)')
    parser.add_argument('--max-gap', type=float, default=0.2)
    parser.add_argument('--window', type=float, default=0.2)
    parser.add_argument('--stills', type=int, default=8, help='annotated still images to save')
    parser.add_argument('--boxes', type=Path, help='person boxes from person_boxes.py (adds trackdet)')
    args = parser.parse_args()
    out = args.out or str(Path('data') / args.video.stem)

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    boxes = ev.load_boxes(args.boxes) if args.boxes else {}
    trackers = ev.TRACKERS + (('trackdet',) if boxes else ())
    shown = trackers[-1] if boxes else 'track'                 # 영상에 그려 보여 줄 방식
    net = ev.MoveNet(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None                                              # 첫 프레임에서 크기를 알고 만든다
    state = {m: dict(region=None, tracking=False) for m in trackers}
    rows, previous_small, i, started = [], None, 0, time.time()
    # 정지 그림을 남길 프레임: 영상을 --stills 등분한 각 칸의 가운데
    still_at = {round(total * (k + 0.5) / args.stills) for k in range(args.stills)}
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(out + '_annotated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (ev.CAM_WIDTH, round(ev.CAM_WIDTH * h / w)),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape
        for st in state.values():
            if st['region'] is None:
                st['region'] = ev.whole_frame(h, w)
        # 카메라 움직임 추정, trackcam 영역을 그만큼 옮기기
        motion = ev.camera_motion(previous_small, small, state['trackcam']['region']) if previous_small is not None else None
        previous_small = small
        row = dict(frame=i, seconds=i / fps)
        if motion is not None:
            if state['trackcam']['tracking']:
                state['trackcam']['region'] = ev.move_region(state['trackcam']['region'], motion[0], sh, sw)
            row.update(cam_dx=motion[0][0, 2] / sw, cam_dy=motion[0][1, 2] / sh,
                       cam_zoom=math.hypot(motion[0][0, 0], motion[0][0, 1]), cam_inliers=motion[1])
        for method, st in state.items():
            # trackdet: 스키어를 놓친 상태에서 검출 상자가 있으면 그 상자에서 다시 시작
            if method == 'trackdet' and not st['tracking'] and ('', '', i) in boxes:
                st['region'], st['tracking'] = ev.box_region(boxes['', '', i], h, w), True
            region, tracking = st['region'], st['tracking']
            xy, score = net.crop(rgb, region)
            found = ev.legs(xy, score)
            row[method + '_left'], row[method + '_right'] = found['left'], found['right']
            row[method + '_tracking'] = int(tracking)
            row[method + '_region_h'] = float(region[2] - region[0])
            row[method + '_cx'], row[method + '_cy'] = (xy[ev.L_HIP] + xy[ev.R_HIP]) / 2 / [w, h]    # 골반 가운데 위치
            # 몸통 길이(어깨~골반, 좌우 평균) ÷ 화면 높이: 스키어가 얼마나 크게 보이는지
            torso = [math.dist(xy[a], xy[b]) for a, b in ((ev.L_SHOULDER, ev.L_HIP), (ev.R_SHOULDER, ev.R_HIP))]
            row[method + '_torso_h'] = float(np.mean(torso)) / h if tracking else None
            row[method + '_minscore'] = float(score[[11, 12, 13, 14, 15, 16]].min())   # 다리 6점 중 가장 낮은 점수
            st['region'], st['tracking'] = ev.next_region(xy, score, h, w)
            if method == shown:                                # 확인용 영상에 그리기
                if tracking:
                    y0, x0, y1, x1 = region
                    cv2.rectangle(bgr, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 255), 2)
                for side, ids in ev.LEGS.items():
                    detected = bool(np.all(score[ids] >= ev.KP_THRESH))
                    if detected or tracking:
                        cv2.polylines(bgr, [xy[ids].astype(np.int32)], False, (0, 0, 255), 3 if detected else 1)
                text = 'L %s  R %s' % tuple('--' if found[s] is None else '%.0f' % found[s] for s in ('left', 'right'))
                # 검은 굵은 글자 위에 흰 글자를 겹쳐 어떤 배경에서도 읽히게 한다
                cv2.putText(bgr, '%.1fs  %s' % (i / fps, text), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(bgr, '%.1fs  %s' % (i / fps, text), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(bgr)
        if i in still_at:
            cv2.imwrite('%s_still_%05d.jpg' % (out, i), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        i += 1
    cap.release()
    writer.release()
    seconds = time.time() - started
    df = pd.DataFrame(rows)

    # 방식별로 원래 값·메운 값·완화한 값의 검출률과 프레임 간 변화량
    max_gap, window = max(1, round(args.max_gap * fps)), max(3, round(args.window * fps) | 1)
    summary = []
    for method in trackers:
        stages = {}
        for side in ('left', 'right'):
            raw = df[f'{method}_{side}'].to_numpy(float)
            filled = fill_short_gaps(raw, max_gap)
            stages[side] = dict(raw=raw, filled=filled, smoothed=smooth(filled, window))
            df[f'{method}_filled_{side}'], df[f'{method}_smoothed_{side}'] = filled, stages[side]['smoothed']
        for stage in ('raw', 'filled', 'smoothed'):
            left, right = stages['left'][stage], stages['right'][stage]
            step = np.concatenate([np.abs(np.diff(left)), np.abs(np.diff(right))])
            summary.append(dict(method=method, stage=stage, frames=len(df),
                                both_pct=100 * np.mean(~np.isnan(left) & ~np.isnan(right)),
                                left_pct=100 * np.mean(~np.isnan(left)), right_pct=100 * np.mean(~np.isnan(right)),
                                whole_frame_pct=100 * (1 - df[method + '_tracking'].mean()),
                                step_deg_per_frame=float(np.nanmean(step)) if np.isfinite(step).any() else float('nan')))
    df.to_csv(out + '_frames.csv', index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out + '_summary.csv', index=False)

    pd.set_option('display.width', 220)
    print('frames %d | %.1f fps | %.1f s | %.1f ms/frame' % (len(df), fps, len(df) / fps, 1000 * seconds / len(df)))
    print(summary.round(2).to_string(index=False))
    # 5초 단위로 나눠 어디서 스키어를 놓쳤는지, 스키어 크기와 카메라 흔들림이 어땠는지 본다
    df['block'] = (df.seconds // 5 * 5).astype(int)
    both = lambda m, s='': df[f'{m}{s}_left'].notna() & df[f'{m}{s}_right'].notna()
    blocks = df.assign(track_both=both('track'), shown_both=both(shown), filled_both=both(shown, '_filled')) \
        .groupby('block').agg(track_both_pct=('track_both', 'mean'), shown_both_pct=('shown_both', 'mean'),
                              shown_filled_pct=('filled_both', 'mean'), shown_tracking_pct=(shown + '_tracking', 'mean'),
                              torso_h_median=(shown + '_torso_h', 'median'), cam_dx=('cam_dx', lambda x: x.abs().mean()),
                              cam_dy=('cam_dy', lambda x: x.abs().mean()))
    blocks[blocks.columns[:4]] *= 100                          # 앞 네 칸은 비율이라 % 로
    print('shown =', shown)
    print('by 5-second block (torso_h = shoulder-to-hip length / frame height while tracking; cam = mean shift per frame):')
    print(blocks.round(3).to_string())


if __name__ == '__main__':
    main()

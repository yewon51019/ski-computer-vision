"""EPFL Ski-2DPose 연속 영상에서 MoveNet Thunder를 평가한다 (영상 기준 평가).

시퀀스 영상(Videos/<영상>/<시퀀스>.mp4)을 처음부터 순서대로 읽으며 MoveNet SinglePose Thunder v4를
여러 방식으로 돌린다.

  full      화면 전체를 256×256으로 줄이고 빈 곳을 채워 넣는다 (movenet_eval.ipynb 사진 평가와 똑같이).
            프레임끼리 서로 상관이 없으므로 정답이 있는 프레임에서만 돌린다.
  track     직전 프레임에서 찾은 관절로 이번 프레임에 넣을 영역을 정한다(MoveNet 공식 예제의 자르기 규칙).
            모든 프레임에서 돈다. 시퀀스 시작과 몸통을 놓쳤을 때는 화면 전체로 돌아간다.
  trackcam  track과 같되, 직전 프레임 → 이번 프레임 사이의 카메라 움직임(배경으로 추정)만큼 영역을 먼저 옮긴다.
  trackdet  track과 같되, 스키어를 놓친 동안에는 사람 검출기가 찾은 상자(person_boxes.py, --boxes)에서 다시 시작한다.

정답이 있는 프레임(ski2dpose_labels.json의 frame_idx)에서 사진 평가와 같은 규칙으로 정답과 비교한다.
  - 한 다리의 골반·무릎·발목 점수가 모두 0.30 이상이면 그 다리를 '찾음'
  - 왼쪽·오른쪽은 모델이 낸 그대로 쓴다(바꿔 맞추지 않음)
  - 무릎 굽힘 = 180° − 화면상 내각

만드는 것 (--out으로 앞 이름 지정)
  <out>_frames.csv      정답 프레임마다 한 줄
  <out>_sequences.csv   시퀀스마다 한 줄. 모든 프레임 기준 검출률(track 계열)
  <out>_summary.csv     방식 × 스키어 크기 묶음별 요약
  <allframes>           모든 프레임: 방식별 무릎 굽힘과 카메라 움직임
                        (movenet_video_post.py의 누락 메우기·흔들림 완화 입력)

실행 (프로젝트 루트, GPU 환경)
  .movenet-gpu\\python.exe movenet_video_eval.py --videos data/ski2dpose_videos/Videos

다른 스크립트(ski_analyze.py, ski24_eval.py)도 이 파일의 MoveNet·자르기 규칙을 가져다 쓴다.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path

# conda 환경은 CUDA·cuDNN DLL을 <환경>/Library/bin에 둔다. TensorFlow는 PATH에서 이 파일들을 찾으므로
# TensorFlow를 불러오기 '전에' 그 폴더를 PATH 앞에 넣는다 (conda activate 없이 실행하기 위해).
_lib_bin = Path(sys.prefix) / 'Library' / 'bin'
if _lib_bin.is_dir():
    os.environ['PATH'] = str(_lib_bin) + os.pathsep + os.environ.get('PATH', '')

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

KP_THRESH = 0.30         # 다리 세 점이 모두 이 점수 이상이어야 '찾음' (사진 평가와 같은 기준)
CROP_MIN_SCORE = 0.2     # 다음 프레임 영역을 정할 때 쓰는 관절 점수 기준 (MoveNet 공식 예제 값)
KNEE_NEAR = 0.5          # 예측 무릎이 정답 무릎에서 몸통 길이 0.5배 안이면 '스키어 위에 있음'으로 본다
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12   # MoveNet(COCO) 점 번호
LEGS = {'left': [11, 13, 15], 'right': [12, 14, 16]}   # COCO 순서의 골반, 무릎, 발목
GT_LEGS = {'right': 10, 'left': 13}                    # EPFL 순서에서 각 다리의 첫 점(골반) 번호. 그 뒤 두 점이 무릎·발목
TRACKERS = ('track', 'trackcam')      # 기본으로 돌릴 추적 방식. --trackers로 바꿀 수 있고 trackdet은 --boxes가 필요
BOX_MARGIN = 0.15        # 검출 상자를 양옆으로 이 비율만큼 넓혀서 넣는다
CAM_WIDTH = 480          # 카메라 움직임은 프레임을 너비 480px로 줄여서 계산한다 (빠르게)


def flexion(points):
    """무릎 굽힘(도) = 180 − 무릎에서의 내각. 다리 한 마디의 길이가 0이면 None.

    points: [골반, 무릎, 발목] 각 (x, y). atan2(외적, 내적)로 각을 구해 0~180° 범위가 안정적으로 나온다.
    """
    hip, knee, ankle = points
    a = (hip[0] - knee[0], hip[1] - knee[1])        # 무릎 → 골반
    b = (ankle[0] - knee[0], ankle[1] - knee[1])    # 무릎 → 발목
    if not math.hypot(*a) or not math.hypot(*b):
        return None
    return 180 - math.degrees(math.atan2(abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1]))


# ------------------------------------------------------------------ 화면에서 모델에 넣을 영역
# 영역은 모두 (y0, x0, y1, x1) 순서, 프레임 대비 비율이다 (TensorFlow crop_and_resize 형식).
def whole_frame(h, w):
    """화면 전체를 정사각형으로 채운 영역. 긴 변에 맞추므로 0~1 밖으로 나갈 수 있다(밖은 검은색으로 채워짐)."""
    if w > h:
        y0 = (h / 2 - w / 2) / h
        return np.array([y0, 0.0, y0 + w / h, 1.0], np.float32)
    x0 = (w / 2 - h / 2) / w
    return np.array([0.0, x0, 1.0, x0 + h / w], np.float32)


def box_region(box, h, w):
    """사람 검출 상자(x0, y0, x1, y1 비율) 둘레의 정사각형 영역. 긴 변 기준, 양옆 BOX_MARGIN만큼 여유."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    half = max((x1 - x0) * w, (y1 - y0) * h) * (1 + 2 * BOX_MARGIN) / 2
    return np.array([(cy - half) / h, (cx - half) / w, (cy + half) / h, (cx + half) / w], np.float32)


def load_boxes(path):
    """person_boxes.py 결과를 {(영상, 시퀀스, 프레임): (x0, y0, x1, y1)}로 읽는다. 사람이 없던 프레임은 뺀다."""
    table = pd.read_csv(path, dtype={'video': str, 'split': str}).fillna({'video': '', 'split': ''})
    table = table[table.conf.notna()]
    return {(r.video, r.split, r.frame): (r.x0, r.y0, r.x1, r.y1) for r in table.itertuples()}


def next_region(xy, score, h, w):
    """이번 프레임의 관절로 다음 프레임에 넣을 영역을 정한다.

    MoveNet 공식 예제의 자르기 규칙: 양쪽 골반 가운데를 중심으로, 몸통 크기의 1.9배와
    확실한 관절들이 퍼진 범위의 1.2배 중 큰 쪽을 반지름으로 한 정사각형.
    골반이나 어깨를 하나도 확실히 못 찾았으면 화면 전체로 돌아간다.
    돌려주는 값: (영역, tracking). 화면 전체면 tracking = False.
    """
    seen = score > CROP_MIN_SCORE
    if not ((seen[L_HIP] or seen[R_HIP]) and (seen[L_SHOULDER] or seen[R_SHOULDER])):
        return whole_frame(h, w), False
    centre = (xy[L_HIP] + xy[R_HIP]) / 2
    torso_x, torso_y = np.abs(xy[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] - centre).max(0)   # 몸통이 중심에서 퍼진 정도
    body_x, body_y = np.abs(xy[score >= CROP_MIN_SCORE] - centre).max(0)                     # 확실한 관절 전체가 퍼진 정도
    half = max(torso_x * 1.9, torso_y * 1.9, body_x * 1.2, body_y * 1.2)
    cx, cy = centre
    half = min(half, max(cx, w - cx, cy, h - cy))      # 화면 끝까지보다 더 크게 잡지 않는다
    if half > max(w, h) / 2:                           # 화면보다 크면 그냥 화면 전체
        return whole_frame(h, w), False
    return np.array([(cy - half) / h, (cx - half) / w, (cy + half) / h, (cx + half) / w], np.float32), True


# ------------------------------------------------------------------ 카메라 움직임
def camera_motion(previous, current, region):
    """직전 프레임 → 이번 프레임의 배경 움직임 (둘 다 작게 줄인 회색 그림).

    region(스키어 영역) 밖의 모서리 점들을 광류(optical flow)로 따라가고, RANSAC으로
    이동·확대·회전을 합친 변환(similarity)을 맞춘다. 스키어 자신의 움직임이 섞이지 않게 스키어 영역은 뺀다.
    돌려주는 값: (작은 그림 픽셀 기준 2×3 행렬, 맞은 점 수). 점이 8개 미만이면 None(추정 불가).
    """
    sh, sw = previous.shape
    mask = np.full((sh, sw), 255, np.uint8)
    y0, x0, y1, x1 = region
    if y1 - y0 < 1.0:                              # 화면 전체가 아니라 실제로 자른 영역일 때만 스키어를 뺀다
        mask[max(0, int(y0 * sh)):max(0, int(y1 * sh)), max(0, int(x0 * sw)):max(0, int(x1 * sw))] = 0
    points = cv2.goodFeaturesToTrack(previous, 200, 0.01, 8, mask=mask)     # 따라가기 좋은 모서리 점 최대 200개
    if points is None or len(points) < 8:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    good = status.ravel() == 1                     # 이번 프레임에서도 찾은 점만
    if good.sum() < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                                  ransacReprojThreshold=2.0)
    if matrix is None:
        return None
    return matrix, int(inliers.sum())


def move_region(region, matrix, sh, sw):
    """카메라 움직임 행렬을 영역(비율)에 적용한다: 가운데를 옮기고 확대 비율만큼 크기를 바꾼다."""
    y0, x0, y1, x1 = region
    cx, cy = matrix @ np.array([(x0 + x1) / 2 * sw, (y0 + y1) / 2 * sh, 1.0])
    zoom = math.hypot(matrix[0, 0], matrix[0, 1])
    half_x, half_y = (x1 - x0) / 2 * zoom, (y1 - y0) / 2 * zoom
    return np.array([cy / sh - half_y, cx / sw - half_x, cy / sh + half_y, cx / sw + half_x], np.float32)


# ------------------------------------------------------------------ 모델
class MoveNet:
    """MoveNet SinglePose Thunder v4 (SavedModel). full·crop 모두 (프레임 픽셀 좌표 17×2, 점수 17개)를 돌려준다."""

    def __init__(self, model_dir):
        self.model = tf.saved_model.load(str(model_dir))
        infer = self.model.signatures['serving_default']

        # tf.function: 그래프로 한 번 만들어 두고 매 프레임 빠르게 실행한다. 입력 모양을 고정해 다시 만들지 않게 한다.
        @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8),
                                      tf.TensorSpec([2], tf.int32), tf.TensorSpec([2], tf.int32)])
        def full(frame, size, offset):       # movenet_eval.ipynb와 같은 순서: 줄이기 → 256×256에 채워 넣기
            resized = tf.image.resize(frame, size)
            padded = tf.image.pad_to_bounding_box(resized, offset[0], offset[1], 256, 256)
            return infer(input=tf.cast(padded[None], tf.int32))['output_0'][0, 0]

        @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8),
                                      tf.TensorSpec([4], tf.float32)])
        def crop(frame, region):             # 영역을 잘라 256×256으로 바꿔 넣기 (영역이 화면 밖이면 검은색)
            patch = tf.image.crop_and_resize(tf.cast(frame[None], tf.float32), region[None], [0], [256, 256])
            return infer(input=tf.cast(patch, tf.int32))['output_0'][0, 0]

        self._full, self._crop = full, crop

    def full(self, rgb):
        """화면 전체를 넣는다. 모델 출력(256 기준 y, x 비율)을 원래 프레임 픽셀로 되돌린다."""
        h, w = rgb.shape[:2]
        scale = 256 / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        x0, y0 = (256 - nw) // 2, (256 - nh) // 2       # 가운데 정렬을 위한 여백
        out = self._full(rgb, [nh, nw], [y0, x0]).numpy()
        xy = np.stack([(out[:, 1] * 256 - x0) * w / nw, (out[:, 0] * 256 - y0) * h / nh], 1)
        return xy, out[:, 2]

    def crop(self, rgb, region):
        """영역(y0, x0, y1, x1 비율)만 넣는다. 모델 출력(영역 대비 비율)을 프레임 픽셀로 되돌린다."""
        h, w = rgb.shape[:2]
        out = self._crop(rgb, region).numpy()
        y0, x0, y1, x1 = region
        xy = np.stack([(x0 + out[:, 1] * (x1 - x0)) * w, (y0 + out[:, 0] * (y1 - y0)) * h], 1)
        return xy, out[:, 2]


def legs(xy, score):
    """다리별 무릎 굽힘 {'left': 도, 'right': 도}. 그 다리 세 점 중 하나라도 기준 미만이면 None."""
    return {side: flexion(xy[ids]) if np.all(score[ids] >= KP_THRESH) else None for side, ids in LEGS.items()}


# ------------------------------------------------------------------ 정답
def ground_truth(labels, videos_dir):
    """정답 프레임마다 dict 하나. 정의는 movenet_eval.ipynb와 같다.

    size = 몸통 길이(어깨~골반, 좌우 평균) ÷ 화면 높이. 전체 중앙값(cut)보다 작으면 small, 크면 large 묶음.
    돌려주는 값: (프레임 목록, 중앙값)
    """
    rows = []
    for video, splits in labels.items():
        for split, records in splits.items():
            # 정답 좌표는 비율이라 픽셀로 바꾸려면 영상 크기가 필요하다
            cap = cv2.VideoCapture(str(videos_dir / video / f'{split}.mp4'))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            for image, record in records.items():
                p = [(x * w, y * h, v) for x, y, v in record['annotation']]
                row = dict(video=video, split=split, image=f'Images_webp/{video}/{split}/{image}.webp',
                           frame_idx=record['frame_idx'], width=w, height=h, fps=fps, points=p)
                # EPFL 2(오른어깨)–10(오른골반), 6(왼어깨)–13(왼골반) 중 보이는 쪽으로 몸통 길이
                sizes = [math.dist(p[a][:2], p[b][:2]) / h for a, b in [(2, 10), (6, 13)] if p[a][2] and p[b][2]]
                row['size'] = statistics.mean(sizes) if sizes else None
                for side, start in GT_LEGS.items():
                    leg = p[start:start + 3]
                    row['gt_' + side] = flexion(leg) if all(j[2] for j in leg) else None   # 세 점이 다 보일 때만 정답 각도
                rows.append(row)
    cut = statistics.median(r['size'] for r in rows if r['size'] is not None)
    for r in rows:
        r['group'] = 'unknown' if r['size'] is None else ('small' if r['size'] <= cut else 'large')
    return rows, cut


def knee_error(row, xy, side):
    """예측 무릎과 정답 무릎의 거리 (몸통 길이 단위). 정답 무릎이 안 보이거나 몸통 길이를 모르면 None."""
    knee = row['points'][GT_LEGS[side] + 1]
    if not knee[2] or row['size'] is None:
        return None
    return math.dist(xy[LEGS[side][1]], knee[:2]) / (row['size'] * row['height'])


# ------------------------------------------------------------------ 보조 함수
def thumb(bgr):
    """프레임 번호 맞춤 확인용 아주 작은 회색 그림(96×54). 영상 프레임과 정답 사진을 비교하는 데 쓴다."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)


def draw_debug(path, bgr, row, results, region):
    """확인용 그림 저장: 초록 = 정답 다리, 노랑 = full, 빨강 = 마지막 추적 방식, 흰 상자 = 그 방식의 영역.

    '찾음' 기준을 넘은 다리는 굵게, 못 넘은 다리는 가늘게 그린다.
    """
    img = bgr.copy()
    h, w = img.shape[:2]
    y0, x0, y1, x1 = region
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 255), 2)
    for start in GT_LEGS.values():
        leg = row['points'][start:start + 3]
        if all(j[2] for j in leg):
            cv2.polylines(img, [np.array([j[:2] for j in leg], np.int32)], False, (0, 255, 0), 3)
    for method, colour in (('full', (0, 255, 255)), (list(results)[-2], (0, 0, 255))):
        xy, score = results[method]
        for ids in LEGS.values():
            ok = bool(np.all(score[ids] >= KP_THRESH))
            cv2.polylines(img, [xy[ids].astype(np.int32)], False, colour, 3 if ok else 1)
    scale = 1280 / w                                   # 저장 그림은 너비 1280으로 맞춘다
    cv2.imwrite(str(path), cv2.resize(img, (1280, round(h * scale))), [cv2.IMWRITE_JPEG_QUALITY, 85])


def summarize(df, method):
    """한 방식의 요약: 전체·크기 묶음별로 양쪽 무릎 검출률, 좌우 각도 MAE, 무릎 위치가 스키어 위인 비율."""
    rows = []
    for name, g in [('all', df), *df.groupby('group')]:
        gt_both = g[['gt_left', 'gt_right']].notna().all(axis=1)                     # 정답에 양쪽 각도가 다 있음
        pred_both = g[[f'{method}_left', f'{method}_right']].notna().all(axis=1)     # 모델이 양쪽 다 찾음
        r = dict(method=method, group=name, frames=len(g), both_pred=int(pred_both.sum()),
                 coverage_pct=100 * pred_both.mean(), both_gt=int(gt_both.sum()),
                 both_paired=int((gt_both & pred_both).sum()))
        for side in ('left', 'right'):
            errors = (g['gt_' + side] - g[f'{method}_{side}']).abs().dropna()      # 둘 다 있는 프레임만 남음
            r[side + '_mae_deg'] = errors.mean()
            r[side + '_pairs'] = len(errors)
        knee = pd.concat([g[f'{method}_kneeerr_left'], g[f'{method}_kneeerr_right']]).dropna()
        r['knee_near_pct'] = 100 * (knee <= KNEE_NEAR).mean() if len(knee) else float('nan')
        r['knee_checked'] = len(knee)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 본 처리
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos', type=Path, required=True, help='folder holding <video>/<split>.mp4')
    parser.add_argument('--labels', type=Path, default=Path('ski2dpose_labels.json.zip'))
    parser.add_argument('--images', type=Path, default=Path('ski2dpose_images_webp.zip'),
                        help='labelled photos, used only to confirm that frame numbers line up')
    parser.add_argument('--model', type=Path, default=Path('movenet-tensorflow2-singlepose-thunder-v4'))
    parser.add_argument('--out', default='movenet_video')
    parser.add_argument('--allframes', default='data/movenet_video_allframes.csv',
                        help='per-frame output for every frame (large, kept out of Git)')
    parser.add_argument('--limit', type=int, default=0, help='only the first N sequences (trial run)')
    parser.add_argument('--boxes', type=Path, help='person boxes from person_boxes.py (enables trackdet)')
    parser.add_argument('--trackers', default=','.join(TRACKERS), help='comma list of track, trackcam, trackdet')
    parser.add_argument('--debug-dir', type=Path, help='save annotated labelled frames here')
    parser.add_argument('--debug-every', type=int, default=1, help='save every Nth labelled frame')
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)      # GPU 메모리는 필요한 만큼만
    print('TensorFlow', tf.__version__, '| GPU', [g.name for g in gpus] or 'none (CPU)', flush=True)

    # 정답 읽기 → 시퀀스별로 {프레임 번호: 정답} 묶기
    with zipfile.ZipFile(args.labels) as archive:
        labels = json.loads(archive.read('ski2dpose_labels.json'))
    rows, cut = ground_truth(labels, args.videos)
    print('labelled frames', len(rows), '| torso / image height median %.4f' % cut, flush=True)
    by_seq = {}
    for r in rows:
        by_seq.setdefault((r['video'], r['split']), {})[r['frame_idx']] = r
    sequences = list(by_seq)[:args.limit or None]
    photos = zipfile.ZipFile(args.images) if args.images.is_file() else None    # 프레임 번호 맞춤 확인용
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    trackers = tuple(args.trackers.split(','))
    methods = ('full',) + trackers
    boxes = load_boxes(args.boxes) if args.boxes else {}
    assert 'trackdet' not in trackers or boxes, 'trackdet needs --boxes'
    # 카메라 움직임 계산 때 '스키어 영역'으로 뺄 영역은 trackcam(없으면 첫 추적 방식)의 영역을 쓴다
    cam_user = 'trackcam' if 'trackcam' in trackers else trackers[0]
    net = MoveNet(args.model)
    seq_rows, all_rows, saved, started = [], [], 0, time.time()
    for number, (video, split) in enumerate(sequences, 1):
        want = by_seq[video, split]                    # 이 시퀀스의 정답 프레임들
        cap = cv2.VideoCapture(str(args.videos / video / f'{split}.mp4'))
        # 추적 방식마다 따로 상태를 둔다: 다음 영역, 추적 중 여부, 검출 프레임 수 집계
        state = {m: dict(region=None, tracking=False, both=0, left=0, right=0, whole=0) for m in trackers}
        n_frames, cam_ok, i, t0 = 0, 0, 0, time.time()
        previous_thumb, waiting, previous_small = None, None, None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (CAM_WIDTH, round(CAM_WIDTH * h / w)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape
            for st in state.values():
                if st['region'] is None:
                    st['region'] = whole_frame(h, w)
            # 1) 카메라 움직임 추정 (첫 프레임은 비교할 앞 프레임이 없음)
            motion = None
            if previous_small is not None:
                motion = camera_motion(previous_small, small, state[cam_user]['region'])
            previous_small = small
            frame_out = dict(video=video, split=split, frame=i)
            if motion is None:
                frame_out.update(cam_dx=None, cam_dy=None, cam_zoom=None, cam_inliers=0)
            else:                                   # 이동량은 프레임 너비·높이 대비 비율로 기록
                cam_ok += 1
                if 'trackcam' in state and state['trackcam']['tracking']:
                    state['trackcam']['region'] = move_region(state['trackcam']['region'], motion[0], sh, sw)
                frame_out.update(cam_dx=motion[0][0, 2] / sw, cam_dy=motion[0][1, 2] / sh,
                                 cam_zoom=math.hypot(motion[0][0, 0], motion[0][0, 1]), cam_inliers=motion[1])
            # 2) 추적 방식마다 MoveNet 실행 → 다리 각도 기록 → 다음 영역 결정
            results, used = {}, {}
            for method, st in state.items():
                if method == 'trackdet' and not st['tracking'] and (video, split, i) in boxes:
                    st['region'], st['tracking'] = box_region(boxes[video, split, i], h, w), True
                used[method] = (st['region'], st['tracking'])      # 이번 프레임에 실제로 쓴 영역 (기록용)
                xy, score = net.crop(rgb, st['region'])
                results[method] = (xy, score)
                found = legs(xy, score)
                st['whole'] += not st['tracking']
                st['left'] += found['left'] is not None
                st['right'] += found['right'] is not None
                st['both'] += found['left'] is not None and found['right'] is not None
                frame_out[method + '_left'], frame_out[method + '_right'] = found['left'], found['right']
                frame_out[method + '_tracking'] = int(st['tracking'])
                st['region'], st['tracking'] = next_region(xy, score, h, w)
            all_rows.append(frame_out)
            n_frames += 1

            # 3) 프레임 번호 맞춤 확인: 정답 사진이 '자기 프레임'과 가장 닮았는지 앞뒤 프레임과 비교
            near = photos is not None and (i in want or i - 1 in want or i + 1 in want)
            small_thumb = thumb(bgr) if near else None
            if waiting is not None:               # 정답 프레임 바로 다음 프레임: 다음 프레임과의 차이 기록
                waiting[0]['align_next'] = float(np.abs(small_thumb - waiting[1]).mean())
                waiting = None
            # 4) 정답 프레임이면: full도 돌리고, 모든 방식의 각도·무릎 위치 오차를 기록
            if i in want:
                row = want[i]
                results['full'] = net.full(rgb)
                for method, (mxy, mscore) in results.items():
                    for side, value in legs(mxy, mscore).items():
                        row[f'{method}_{side}'] = value
                        row[f'{method}_kneeerr_{side}'] = knee_error(row, mxy, side) if value is not None else None
                for method in trackers:
                    row[method + '_tracking'] = int(used[method][1])
                    row[method + '_region_h'] = float(used[method][0][2] - used[method][0][0])   # 영역 높이(프레임 높이 대비)
                if photos is not None:
                    photo = thumb(cv2.imdecode(np.frombuffer(photos.read(row['image']), np.uint8), cv2.IMREAD_COLOR))
                    row['align_diff'] = float(np.abs(small_thumb - photo).mean())
                    row['align_prev'] = None if previous_thumb is None else float(np.abs(previous_thumb - photo).mean())
                    waiting = (row, photo)
                if args.debug_dir and saved % args.debug_every == 0:
                    draw_debug(args.debug_dir / f'{video}_{split}_{i:05d}.jpg', bgr, row, results, used[trackers[-1]][0])
                saved += 1
            previous_thumb = small_thumb
            i += 1
        cap.release()
        # 시퀀스 요약 한 줄
        seconds = time.time() - t0
        missing = [k for k in want if k >= n_frames]   # 영상이 정답 프레임 번호까지 가지 못한 경우
        seq_row = dict(video=video, split=split, fps=next(iter(want.values()))['fps'], frames=n_frames,
                       labelled=len(want), labelled_missing=len(missing), camera_estimated=cam_ok,
                       seconds=round(seconds, 2))
        for method, st in state.items():
            seq_row.update({method + '_both': st['both'], method + '_both_pct': 100 * st['both'] / max(1, n_frames),
                            method + '_left': st['left'], method + '_right': st['right'],
                            method + '_whole_frame': st['whole']})
        seq_rows.append(seq_row)
        print('%3d/%d %s/%s frames=%d labelled=%d both: %s  %.1f ms/frame'
              % (number, len(sequences), video, split, n_frames, len(want),
                 ' '.join('%s %.1f%%' % (m, seq_row[m + '_both_pct']) for m in trackers),
                 1000 * seconds / max(1, n_frames)), flush=True)

    # 결과 파일 쓰기
    processed = set(sequences)
    done = [r for r in rows if (r['video'], r['split']) in processed]
    numeric = [f'{m}_{s}' for m in methods for s in ('left', 'right')] + \
              [f'{m}_kneeerr_{s}' for m in methods for s in ('left', 'right')]
    for r in done:                                # 영상이 닿지 못한 정답 프레임은 빈 값으로 남긴다
        for column in numeric + [m + x for m in trackers for x in ('_tracking', '_region_h')]:
            r.setdefault(column, None)
    df = pd.DataFrame(done).drop(columns=['points'])
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df.to_csv(args.out + '_frames.csv', index=False)
    seq = pd.DataFrame(seq_rows)
    seq.to_csv(args.out + '_sequences.csv', index=False)
    Path(args.allframes).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.allframes, index=False)
    summary = pd.DataFrame([r for method in methods for r in summarize(df, method)])
    summary.to_csv(args.out + '_summary.csv', index=False)

    # 화면 출력: 요약표, 모든 프레임 기준 검출률, 카메라 추정 비율, 프레임 번호 맞춤 확인 결과
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(summary.round(2).to_string(index=False))
    total = int(seq.frames.sum())
    for method in trackers:
        print('all frames (%s): %d | both knees %d (%.1f%%) | whole-frame input %d (%.1f%%)'
              % (method, total, seq[method + '_both'].sum(), 100 * seq[method + '_both'].sum() / total,
                 seq[method + '_whole_frame'].sum(), 100 * seq[method + '_whole_frame'].sum() / total))
    print('camera motion estimated on %d of %d frames | %.1f ms/frame (all trackers) | %.1f min'
          % (seq.camera_estimated.sum(), total, 1000 * seq.seconds.sum() / total, (time.time() - started) / 60))
    if 'align_diff' in df:
        worse = (df.align_diff >= df[['align_prev', 'align_next']].min(axis=1)).sum()
        print('frame-number check: labelled frames %d | photo matches its own frame best on %d | not best on %d | '
              'missing in video %d' % (len(df), len(df) - worse - df.align_diff.isna().sum(), worse,
                                        df.align_diff.isna().sum()))


if __name__ == '__main__':
    main()

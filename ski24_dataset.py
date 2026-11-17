"""EPFL Ski-2DPose로 24점(몸·스키·폴) 스키어 관절 학습 데이터를 만든다 (YOLO-Pose 형식).

왜 이렇게 만드나
  - 앱은 화면 전체가 아니라 '고른 스키어 주변을 잘라서' 관절을 찾는다. 그래서 학습 그림도
    정답 프레임마다 스키어 주변을 정사각형으로 잘라 한 장씩 만든다.
  - 같은 영상에서 뽑은 프레임끼리는 거의 똑같다. 프레임 단위로 학습/평가를 섞으면 모델이
    '본 적 있는 장면'으로 평가받아 점수가 부풀려진다. 그래서 '영상 단위'로 나눈다.

만들어지는 것 (data/ski24/)
  images/<묶음>/<프레임 ID>.jpg   잘라 낸 그림. 묶음은 train(학습)·val(검증)·test(평가)
  labels/<묶음>/<프레임 ID>.txt   YOLO-Pose 정답 한 줄:
                                  클래스, 상자(가운데 x·y, 너비, 높이), 그리고 24점마다 x y v
                                  (v = 2 보임, 1 가려졌지만 위치는 표시됨). 모두 잘라 낸 그림 대비 비율(0~1)
  frames.csv                      프레임마다 한 줄: 묶음, 영상, 시퀀스, 프레임 번호, 원본 크기,
                                  잘라 낸 위치(원본 픽셀). 평가 때 예측을 원본 좌표로 되돌리는 데 쓴다
  ski24.yaml                      ultralytics가 읽는 데이터 설정 파일

실행 (OpenCV가 있는 환경, 예: .yolo-venv)
  .yolo-venv\\Scripts\\python.exe ski24_dataset.py
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

# EPFL 정답의 점 순서(24개). 왼쪽·오른쪽은 화면 기준이 아니라 '스키어 자신의' 왼쪽·오른쪽이다.
POINTS = ['head', 'neck',                                                       # 0 머리, 1 목
          'right_shoulder', 'right_elbow', 'right_hand', 'right_pole_tip',      # 2~5 오른팔과 오른쪽 폴 끝
          'left_shoulder', 'left_elbow', 'left_hand', 'left_pole_tip',          # 6~9 왼팔과 왼쪽 폴 끝
          'right_hip', 'right_knee', 'right_ankle',                             # 10~12 오른다리
          'left_hip', 'left_knee', 'left_ankle',                                # 13~15 왼다리
          'right_ski_tip', 'right_toes', 'right_heel', 'right_ski_tail',        # 16~19 오른쪽 스키(앞끝·발끝·뒤꿈치·뒤끝)
          'left_ski_tip', 'left_toes', 'left_heel', 'left_ski_tail']            # 20~23 왼쪽 스키

# 학습 중 그림을 좌우로 뒤집으면(데이터 늘리기) 왼쪽 점과 오른쪽 점이 서로 바뀐다.
# FLIP[i] = 뒤집은 뒤 i번 점이 되어야 하는 점 번호. 머리(0)·목(1)은 가운데라 그대로다.
FLIP = [0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 13, 14, 15, 10, 11, 12, 20, 21, 22, 23, 16, 17, 18, 19]

MARGIN = 1.3                     # 잘라 낼 정사각형 한 변 = 스키어 상자의 긴 변 × 1.3 (주변 여백)
TEST_SHARE, VAL_SHARE = 0.20, 0.10   # 평가용·검증용으로 떼어 둘 프레임 비율(목표치)
SEED = 0                         # 영상 섞는 순서를 고정해 누가 돌려도 같은 분할이 나오게 한다


def load_labels(zip_path):
    """EPFL 정답 zip을 읽어 {영상 ID: [(시퀀스, 프레임 ID, 프레임 번호, 24×3 배열)]}로 돌려준다.

    24×3 배열의 각 줄은 [x, y, 보임 여부]이고 x·y는 원본 프레임 대비 비율(0~1)이다.
    """
    raw = json.loads(zipfile.ZipFile(zip_path).read('ski2dpose_labels.json'))
    videos = {}
    for video, sequences in raw.items():
        videos[video] = [(sequence, frame_id, item['frame_idx'], np.array(item['annotation'], float))
                         for sequence, frames in sequences.items() for frame_id, item in frames.items()]
    return videos


def split_videos(videos):
    """영상을 train / val / test로 나눈다. 돌려주는 값: {영상 ID: 묶음 이름}.

    방법: 고정 시드로 영상 순서를 섞고, 앞에서부터 평가용이 전체 프레임의 20%를 넘을 때까지 채우고,
    다음으로 검증용을 10%까지 채운 뒤 나머지를 학습용으로 둔다. 영상마다 프레임 수가 달라서
    마지막에 들어간 영상 크기만큼 목표 비율을 넘을 수 있다.
    """
    names = sorted(videos)                       # 정렬부터 해서 섞기 전 순서도 항상 같게 한다
    random.Random(SEED).shuffle(names)
    total = sum(len(videos[v]) for v in names)
    sets, counts = {}, {'test': 0, 'val': 0}
    for name in names:
        size = len(videos[name])
        if counts['test'] < TEST_SHARE * total:
            sets[name] = 'test'
        elif counts['val'] < VAL_SHARE * total:
            sets[name] = 'val'
        else:
            sets[name] = 'train'
        counts[sets[name]] = counts.get(sets[name], 0) + size
    return sets


def square_crop(points_px, margin):
    """점들을 감싸는 상자의 긴 변 × margin 크기 정사각형을 [x0, y0, x1, y1](픽셀)로 돌려준다.

    정사각형으로 자르는 이유: 모델 입력이 정사각형이라, 직사각형을 넣으면 늘어나거나 찌그러진다.
    가운데는 상자의 가운데에 맞춘다. 화면 밖으로 나가도 그대로 두고, 자를 때(cut) 회색으로 채운다.
    """
    x0, y0 = points_px.min(0)
    x1, y1 = points_px.max(0)
    side = max(x1 - x0, y1 - y0) * margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


def cut(image, crop):
    """원본 그림에서 crop 영역을 잘라 낸다. 돌려주는 값: (잘라 낸 그림, 정수로 반올림한 crop).

    crop이 화면 밖으로 나간 부분은 YOLO가 빈 곳에 쓰는 회색(114)으로 채운다.
    그래서 스키어가 화면 가장자리에 있어도 항상 같은 크기의 정사각형이 나온다.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in crop]
    h, w = image.shape[:2]
    out = np.full((y1 - y0, x1 - x0, 3), 114, np.uint8)
    # 원본과 crop이 겹치는 부분만 복사한다 (sx·sy = 원본 안으로 잘라 맞춘 경계)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out, (x0, y0, x1, y1)


def label_line(annotation_px, crop):
    """잘라 낸 그림 한 장의 YOLO-Pose 정답 한 줄을 만든다.

    annotation_px: 24×3, [x 픽셀, y 픽셀, 보임 여부] (원본 좌표)
    모든 값은 잘라 낸 그림의 한 변 길이로 나눈 비율로 바꾼다. 상자는 24점을 감싸는 상자다.
    가려진 점(보임 0)도 위치가 표시돼 있으므로 v = 1로 넣어 학습에 쓴다(v = 0이면 무시된다).
    """
    x0, y0, x1, y1 = crop
    side = x1 - x0
    xy = (annotation_px[:, :2] - [x0, y0]) / side
    bx0, by0 = xy.min(0)
    bx1, by1 = xy.max(0)
    fields = [0, (bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0]   # 클래스 0 = 스키어
    for (x, y), visible in zip(xy, annotation_px[:, 2]):
        fields += [x, y, 2 if visible else 1]
    return ' '.join(str(f) if isinstance(f, int) else '%.6f' % f for f in fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', type=Path, default=Path('ski2dpose_labels.json.zip'))
    parser.add_argument('--images', type=Path, default=Path('ski2dpose_images_webp.zip'))
    parser.add_argument('--out', type=Path, default=Path('data/ski24'))
    args = parser.parse_args()

    videos = load_labels(args.labels)
    sets = split_videos(videos)
    images = zipfile.ZipFile(args.images)        # 압축을 풀지 않고 zip 안에서 바로 읽는다
    rows = []
    for video, frames in sorted(videos.items()):
        part = sets[video]
        for folder in ('images', 'labels'):
            (args.out / folder / part).mkdir(parents=True, exist_ok=True)
        for sequence, frame_id, frame_index, annotation in frames:
            # 1) 원본 프레임 읽기 (webp)
            data = np.frombuffer(images.read('Images_webp/%s/%s/%s.webp' % (video, sequence, frame_id)), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            # 2) 정답을 비율에서 픽셀로 바꾸고, 24점을 감싸는 정사각형으로 자르기
            annotation_px = annotation * [w, h, 1]
            picture, crop = cut(image, square_crop(annotation_px[:, :2], MARGIN))
            # 3) 그림과 정답 파일 저장
            cv2.imwrite(str(args.out / 'images' / part / (frame_id + '.jpg')), picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (args.out / 'labels' / part / (frame_id + '.txt')).write_text(label_line(annotation_px, crop) + '\n')
            # 4) 평가 때 원본 좌표로 되돌릴 수 있도록 잘라 낸 위치를 기록
            rows.append(dict(set=part, id=frame_id, video=video, sequence=sequence, frame_idx=frame_index,
                             width=w, height=h, cx0=crop[0], cy0=crop[1], cx1=crop[2], cy1=crop[3]))

    with open(args.out / 'frames.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # ultralytics 데이터 설정: 점 24개 × (x, y, v), 좌우 뒤집기 대응표, 클래스 이름
    (args.out / 'ski24.yaml').write_text(
        'path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n'
        'kpt_shape: [24, 3]\nflip_idx: %s\nnames:\n  0: skier\n' % (args.out.resolve().as_posix(), FLIP))

    # 어떤 영상이 어느 묶음에 들어갔는지 화면에 남긴다 (문서 기록용)
    for part in ('train', 'val', 'test'):
        chosen = [v for v in sorted(videos) if sets[v] == part]
        print('%-5s %4d frames  %s' % (part, sum(len(videos[v]) for v in chosen), ', '.join(chosen)))


if __name__ == '__main__':
    main()

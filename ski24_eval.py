"""24점 스키어 데이터의 '평가용 영상'(ski24_dataset.py의 test 묶음)에서 관절 모델을 평가한다.

두 모델이 서로 다른 Python 환경에 있어서 두 단계로 나눈다.

  predict   모델 하나를 한 가지 잘라 넣기 방식으로 돌려, 찾은 점을 '원본 프레임 좌표'로 저장한다
              --model ski24    학습한 YOLO11m-Pose, 24점 (.yolo-venv)
              --model movenet  MoveNet Thunder v4. EPFL과 겹치는 몸 12점만 (.movenet-gpu)
              --crop truth     정답 점을 감싸는 정사각형(×1.3)으로 자르기. 가장 좋은 경우의 참고값이며 앱에서는 불가능
              --crop detected  사람 검출 상자(data/epfl_boxes.csv)를 감싸는 정사각형(×1.5)으로 자르기. 앱이 실제로 하는 방식
  summary   저장된 예측을 모두 정답과 비교해 표로 보여 준다 (numpy만 있으면 어느 환경이든 가능)

채점 기준
  - '찾음': 점 점수가 모델별 기준 이상 (ski24 0.5, MoveNet 0.3 - 앱과 같은 기준)
  - '맞음'(PCK): 찾았고, 정답과의 거리가 스키어 키의 5% 또는 10% 이내.
    스키어 키 = 정답 24점을 감싸는 상자의 높이(스키·폴 포함)
  - 정답에서 '보임'으로 표시된 점만 채점한다. 못 찾은 점은 틀린 것으로 센다.

실행 (프로젝트 루트에서)
  .yolo-venv\\Scripts\\python.exe ski24_eval.py predict --model ski24 --crop truth
  .yolo-venv\\Scripts\\python.exe ski24_eval.py predict --model ski24 --crop detected
  .movenet-gpu\\python.exe ski24_eval.py predict --model movenet --crop truth
  .movenet-gpu\\python.exe ski24_eval.py predict --model movenet --crop detected
  .yolo-venv\\Scripts\\python.exe ski24_eval.py summary
"""
import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np

from ski24_dataset import MARGIN as TRUTH_MARGIN, POINTS, square_crop

DATA = Path('data/ski24')
DETECTED_MARGIN = 1.5     # 사람 검출 상자는 부츠 근처에서 끝나는 일이 많고 스키·폴은 더 바깥으로 나가서 여백을 더 준다
THRESHOLD = {'ski24': 0.5, 'movenet': 0.3}    # 모델별 '찾음' 점수 기준

# MoveNet 점 번호(COCO 순서) → EPFL 점 번호. 두 모델에 모두 있는 12점(어깨·팔꿈치·손목·골반·무릎·발목)
MOVENET_TO_EPFL = {5: 6, 6: 2, 7: 7, 8: 3, 9: 8, 10: 4, 11: 13, 12: 10, 13: 14, 14: 11, 15: 15, 16: 12}

# 결과 표에서 묶어 보여 줄 점 무리 (번호는 ski24_dataset.POINTS 순서)
GROUPS = {
    'body (12 shared with MoveNet)': sorted(MOVENET_TO_EPFL.values()),   # 몸 12점 - MoveNet과 직접 비교
    'head and neck': [0, 1],                                             # 머리·목 - MoveNet에는 같은 점이 없음
    'pole tips': [5, 9],                                                 # 폴 끝
    'ski tips and tails': [16, 19, 20, 23],                              # 스키 앞끝·뒤끝
    'toes and heels': [17, 18, 21, 22],                                  # 발끝·뒤꿈치
}
KNEES = {'right': (10, 11, 12), 'left': (13, 14, 15)}     # 무릎 각도에 쓰는 세 점: 골반, 무릎, 발목


# ------------------------------------------------------------------ 입력 읽기
def test_frames():
    """frames.csv에서 평가용(test) 프레임 줄만 돌려준다."""
    with open(DATA / 'frames.csv') as f:
        return [row for row in csv.DictReader(f) if row['set'] == 'test']


def annotations(zip_path='ski2dpose_labels.json.zip'):
    """정답 전체를 {프레임 ID: 24×3 배열 [x, y, 보임 여부]}로 돌려준다. x·y는 원본 대비 비율(0~1)."""
    raw = json.loads(zipfile.ZipFile(zip_path).read('ski2dpose_labels.json'))
    return {frame_id: np.array(item['annotation'], float)
            for sequences in raw.values() for frames in sequences.values() for frame_id, item in frames.items()}


def detected_boxes(path='data/epfl_boxes.csv'):
    """person_boxes.py가 저장한 사람 검출 상자: {(영상, 시퀀스, 프레임 번호): [x0, y0, x1, y1] 비율}.

    상자가 비어 있는 줄(그 프레임에서 사람을 못 찾음)은 빼고 읽는다.
    """
    with open(path) as f:
        return {(r['video'], r['split'], r['frame']): [float(r[k]) for k in ('x0', 'y0', 'x1', 'y1')]
                for r in csv.DictReader(f) if r['x0']}


def crop_for(row, kind, boxes, truth):
    """평가할 프레임 하나를 어디서 자를지 정한다. 돌려주는 값: 정사각형 [x0, y0, x1, y1](픽셀) 또는 None.

    kind == 'truth'   : 학습 데이터와 똑같이 정답 24점 상자 × 1.3
    kind == 'detected': 사람 검출 상자 × 1.5. 그 프레임에서 사람을 못 찾았으면 None(점 전부 못 찾음 처리)
    """
    w, h = int(row['width']), int(row['height'])
    if kind == 'truth':
        return square_crop(truth[row['id']][:, :2] * [w, h], TRUTH_MARGIN)
    box = boxes.get((row['video'], row['sequence'], row['frame_idx']))
    if box is None:
        return None
    corners = np.array([[box[0] * w, box[1] * h], [box[2] * w, box[3] * h]])   # 상자 두 꼭짓점(픽셀)
    return square_crop(corners, DETECTED_MARGIN)


# ------------------------------------------------------------------ 모델
class Ski24:
    """학습한 24점 YOLO11m-Pose. points()는 (잘라 낸 그림 기준 24×2 좌표, 24개 점수)를 돌려준다."""

    def __init__(self, weights):
        from ultralytics import YOLO          # 이 환경(.yolo-venv)에서만 있으므로 쓸 때 불러온다
        self.model = YOLO(weights)

    def points(self, picture):
        # conf=0.1: 스키어 상자 후보를 낮은 확신도까지 받아, 아예 못 찾는 경우를 줄인다
        result = self.model.predict(picture, imgsz=512, conf=0.1, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return np.zeros((24, 2)), np.zeros(24)          # 아무도 못 찾음: 모든 점 점수 0
        best = int(result.boxes.conf.argmax())              # 잘라 낸 그림에는 스키어가 한 명이므로 가장 확실한 하나만 쓴다
        return result.keypoints.xy[best].cpu().numpy(), result.keypoints.conf[best].cpu().numpy()


class MoveNetOnCrop:
    """같은 잘라 낸 그림에 MoveNet Thunder를 돌려 24점 배열 자리에 옮겨 담는다. MoveNet에 없는 점은 점수 0."""

    def __init__(self, model_dir):
        import movenet_video_eval as ev                         # CUDA 경로 설정과 TensorFlow 준비를 함께 한다
        import tensorflow as tf
        # GPU 메모리를 처음에 다 잡지 않고 필요한 만큼만 쓴다 (학습과 GPU를 나눠 쓸 수 있게)
        # 참고: 이 환경의 TensorFlow는 GPU를 숨긴 CPU 전용 모드에서 시작하다 죽는다(2026-11-17 확인)
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        self.net = ev.MoveNet(model_dir)

    def points(self, picture):
        whole = np.array([0, 0, 1, 1], np.float32)              # 그림 자체가 이미 잘라 낸 것이므로 전체를 넣는다
        xy, score = self.net.crop(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), whole)
        points, scores = np.zeros((24, 2)), np.zeros(24)
        for coco, epfl in MOVENET_TO_EPFL.items():
            points[epfl], scores[epfl] = xy[coco], score[coco]
        return points, scores


def predict(args):
    """평가용 프레임마다: 자르기 → 모델 → 원본 좌표로 되돌려 CSV(pred_<모델>_<자르기>.csv)에 저장."""
    from ski24_dataset import cut
    model = Ski24(args.weights) if args.model == 'ski24' else MoveNetOnCrop(args.movenet)
    truth, boxes = annotations(), detected_boxes()
    images = zipfile.ZipFile('ski2dpose_images_webp.zip')
    out_rows = []
    for row in test_frames():
        crop = crop_for(row, args.crop, boxes, truth)
        if crop is None:
            continue                                            # 사람을 못 찾은 프레임: 저장하지 않으면 채점 때 '전부 못 찾음'이 된다
        data = np.frombuffer(images.read('Images_webp/%s/%s/%s.webp' % (row['video'], row['sequence'], row['id'])), np.uint8)
        picture, (x0, y0, x1, y1) = cut(cv2.imdecode(data, cv2.IMREAD_COLOR), crop)
        points, scores = model.points(picture)
        for joint, ((x, y), score) in enumerate(zip(points, scores)):
            # 잘라 낸 그림 좌표 + 자른 위치 = 원본 프레임 좌표
            out_rows.append(dict(id=row['id'], joint=joint, x=round(x0 + x, 1), y=round(y0 + y, 1), score=round(float(score), 3)))
    path = DATA / ('pred_%s_%s.csv' % (args.model, args.crop))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'joint', 'x', 'y', 'score'])
        writer.writeheader()
        writer.writerows(out_rows)
    print('saved', path, len(out_rows) // 24, 'frames')


# ------------------------------------------------------------------ 채점
def flexion(hip, knee, ankle):
    """무릎 굽힘(도) = 180 − (골반-무릎-발목이 이루는 각). 다리를 쭉 펴면 0에 가깝다."""
    a, b = hip - knee, ankle - knee
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)    # 1e-9: 길이 0일 때 나눗셈 오류 방지
    return 180 - math.degrees(math.acos(np.clip(cosine, -1, 1)))


def load_predictions(path):
    """predict가 저장한 CSV를 {프레임 ID: (24×2 좌표, 24개 점수)}로 읽는다."""
    frames = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            points, scores = frames.setdefault(r['id'], (np.zeros((24, 2)), np.zeros(24)))
            j = int(r['joint'])
            points[j], scores[j] = (float(r['x']), float(r['y'])), float(r['score'])
    return frames


def score_run(frames, predictions, truth, threshold):
    """한 실행(모델 × 자르기)을 채점한다.

    돌려주는 값
      per_joint: 24×4 배열. 점마다 [정답에서 보임인 수, 그중 찾은 수, 5% 이내 수, 10% 이내 수]
      knees    : {(프레임 ID, 'left'/'right'): 무릎 각도 오차(도)}. 정답 세 점이 다 보이고 모델도 세 점을 다 찾은 다리만
    """
    per_joint = np.zeros((24, 4))
    knees = {}
    for row in frames:
        w, h = int(row['width']), int(row['height'])
        annotation = truth[row['id']] * [w, h, 1]               # 비율 → 픽셀
        height = np.ptp(annotation[:, 1])                       # 스키어 키 = 정답 점들의 세로 범위
        points, scores = predictions.get(row['id'], (np.zeros((24, 2)), np.zeros(24)))   # 예측이 없으면 전부 못 찾음
        found = scores >= threshold
        error = np.linalg.norm(points - annotation[:, :2], axis=1) / height                # 키 대비 거리
        visible = annotation[:, 2] == 1
        per_joint[:, 0] += visible
        per_joint[:, 1] += visible & found
        per_joint[:, 2] += visible & found & (error <= 0.05)
        per_joint[:, 3] += visible & found & (error <= 0.10)
        for side, ids in KNEES.items():
            ids = list(ids)
            if visible[ids].all() and found[ids].all():
                knees[(row['id'], side)] = abs(flexion(*points[ids]) - flexion(*annotation[ids, :2]))
    return per_joint, knees


def summary(args):
    """저장된 예측 파일을 모두 채점해 표 세 개를 출력한다: 점 무리별, 점별, 무릎 각도 오차."""
    frames, truth = test_frames(), annotations()
    runs = {}
    for model in ('ski24', 'movenet'):
        for crop in ('truth', 'detected'):
            path = DATA / ('pred_%s_%s.csv' % (model, crop))
            if path.exists():                                   # 아직 안 돌린 조합은 건너뛴다
                runs[(model, crop)] = score_run(frames, load_predictions(path), truth, THRESHOLD[model])
    print('test frames:', len(frames))

    # 표 1: 점 무리별 찾은 비율과 PCK
    print('\nPCK = found and within 5% / 10% of the skier height, out of points annotated as visible')
    print('%-32s' % 'group', *['%-24s' % ('%s / %s' % key) for key in runs])
    for name, ids in GROUPS.items():
        cells = []
        for per_joint, _ in runs.values():
            n, found, p5, p10 = per_joint[ids].sum(0)
            # 이 모델이 그 무리를 아예 내지 않으면(MoveNet의 스키 등) '-'로 표시
            cells.append('%-24s' % ('-' if found == 0 else 'found %2.0f%% pck %2.0f/%2.0f%%' % (100 * found / n, 100 * p5 / n, 100 * p10 / n)))
        print('%-32s' % name, *cells)

    # 표 2: 점 하나하나의 10% 이내 비율
    print('\nper point (PCK 10%)')
    for j, name in enumerate(POINTS):
        print('%-16s' % name, *['%5.0f%%' % (100 * per_joint[j, 3] / max(1, per_joint[j, 0])) for per_joint, _ in runs.values()])

    # 표 3: 무릎 각도 오차. 모델마다 잰 다리가 달라 공정하게 보려고 '둘 다 잰 다리'끼리도 비교한다
    print('\nknee bend error (degrees) - each model on its own measured legs, then on legs both measured')
    for crop in ('truth', 'detected'):
        if ('ski24', crop) not in runs or ('movenet', crop) not in runs:
            continue
        a, b = runs[('ski24', crop)][1], runs[('movenet', crop)][1]
        both = sorted(set(a) & set(b))
        describe = lambda e: 'mean %.1f median %.1f' % (np.mean(e), np.median(e)) if len(e) else '-'
        print('%-9s ski24 %d legs %s | movenet %d legs %s | both %d legs: ski24 %s, movenet %s' % (
            crop, len(a), describe(list(a.values())), len(b), describe(list(b.values())), len(both),
            describe([a[k] for k in both]), describe([b[k] for k in both])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='step', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--model', choices=['ski24', 'movenet'], required=True)
    p.add_argument('--crop', choices=['truth', 'detected'], required=True)
    p.add_argument('--weights', default='data/ski24/runs/yolo11m_pose_512/weights/best.pt')
    p.add_argument('--movenet', default='movenet-tensorflow2-singlepose-thunder-v4')
    sub.add_parser('summary')
    args = parser.parse_args()
    predict(args) if args.step == 'predict' else summary(args)


if __name__ == '__main__':
    main()

"""EPFL image evaluation. Run beside ski_pose.py and the two original ZIPs."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from ski_pose import knee_angle, webp_size


def estimate(detector, photo):
    import cv2
    import mediapipe as mp
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(photo, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Could not decode WebP')
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    poses = detector.detect(image).pose_landmarks
    if not poses:
        return []
    height, width = frame.shape[:2]
    return [(p.x * width, p.y * height, min(p.visibility, p.presence) >= 0.5) for p in poses[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Optional MediaPipe Pose Landmarker .task file')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('ski_comparison.csv' if args.model else 'ski_angles.csv')
    with ZipFile('ski2dpose_labels.json.zip') as labels:
        name, = [n for n in labels.namelist() if n.endswith('.json') and '__MACOSX/' not in n]
        data = json.loads(labels.read(name))
    fields = ['video_id', 'split_id', 'image_id', 'frame_idx', 'width', 'height', 'visible_points']
    for side in ('right', 'left'):
        fields.extend([side + '_status', side + '_angle_deg'])
        if args.model:
            fields.extend(side + suffix for suffix in ('_model_status', '_model_deg', '_abs_error_deg', '_knee_error_px'))
    statuses = Counter()
    availability = Counter()
    angles = {'right': [], 'left': []}
    errors = {'right': [], 'left': []}
    with ExitStack() as stack:
        detector = None
        if args.model:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.model)))
            detector = stack.enter_context(vision.PoseLandmarker.create_from_options(options))
        images = stack.enter_context(ZipFile('ski2dpose_images_webp.zip'))
        temp = stack.enter_context(TemporaryDirectory(dir=output.parent))
        pending = Path(temp) / 'output.csv'
        with pending.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for video, splits in data.items():
                for split, records in splits.items():
                    for image, record in records.items():
                        photo = images.read(f'Images_webp/{video}/{split}/{image}.webp')
                        width, height = webp_size(photo)
                        points = [(x * width, y * height, v) for x, y, v in record['annotation']]
                        if len(points) != 24 or not all(math.isfinite(x) and math.isfinite(y) and v in (0, 1) for x,y,v in points):
                            raise ValueError(f'Invalid annotation: {image}')
                        row = dict(video_id=video, split_id=split, image_id=image, frame_idx=record['frame_idx'],
                                   width=width, height=height, visible_points=sum(p[2] for p in points))
                        model_points = estimate(detector, photo) if detector else []
                        measured = []
                        for side, start, indices in [('right', 10, (24,26,28)), ('left', 13, (23,25,27))]:
                            angle = knee_angle(points, start)
                            status = 'ok' if angle is not None else 'occluded' if not all(p[2] for p in points[start:start+3]) else 'degenerate'
                            row[side + '_status'] = status
                            statuses[(side, status)] += 1
                            if angle is not None:
                                row[side + '_angle_deg'] = round(angle, 3)
                                angles[side].append(angle)
                                measured.append(side)
                            if detector:
                                leg = [model_points[i] for i in indices] if model_points else []
                                pred = knee_angle(leg, 0) if leg else None
                                row[side + '_model_status'] = 'no_pose' if not leg else 'unmeasurable' if pred is None else 'ok'
                                if leg and leg[1][2] and points[start+1][2]:
                                    row[side + '_knee_error_px'] = round(math.dist(leg[1][:2], points[start+1][:2]), 3)
                                if pred is not None:
                                    row[side + '_model_deg'] = round(pred, 3)
                                    if angle is not None:
                                        errors[side].append(abs(pred-angle))
                                        row[side + '_abs_error_deg'] = round(errors[side][-1], 3)
                        availability[tuple(measured)] += 1
                        writer.writerow(row)
        pending.replace(output)
    print('Videos:', len(data), 'Sequences:', sum(map(len, data.values())), 'Images:', availability.total())
    print('Measurement availability:', dict(availability), 'Joint status:', dict(statuses))
    for side, values in angles.items():
        print(side, 'median:', round(statistics.median(values), 2))
        if errors[side]:
            print(side, 'paired samples:', len(errors[side]), 'MAE:', round(statistics.mean(errors[side]), 2))


if __name__ == '__main__':
    main()

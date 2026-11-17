"""웹 분석 1단계: 영상 속 모든 사람을 찾고, 프레임 사이에서 같은 사람끼리 이어 붙인다(추적).

PyTorch 환경(.yolo-venv)에서 돈다. 사람 검출은 YOLO11m, 이어 붙이기는 ByteTrack.
--out-dir에 만드는 것
  tracks.csv        프레임, 추적 번호, x0, y0, x1, y1(프레임 대비 비율), 확신도 - 2단계가 고른 사람의 상자를 여기서 읽는다
  candidates.json   사용자가 고를 수 있는 후보. 상자 높이 --min-px 이상으로 --min-seconds 이상 보인 사람만,
                    그림 파일 이름과 함께
  thumb_<번호>.jpg  후보 그림: 화면 안에 다 들어온 순간 중 상자가 가장 컸을 때
  progress.json     {"stage": "people", "done": n, "total": N}   (--progress를 줬을 때, 화면 진행 막대용)

실행
  .yolo-venv\\Scripts\\python.exe person_tracks.py clip.mp4 --out-dir data/jobs/clip
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

EDGE = 0.01       # 상자가 화면 가장자리 1% 안쪽까지 닿으면 '잘렸다'고 보고 분석·그림에 쓰지 않는다


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--weights', default='data/models/yolo11m.pt')
    parser.add_argument('--imgsz', type=int, default=1280)         # 검출 입력 크기. 멀리 있는 작은 사람도 잡으려고 크게
    parser.add_argument('--conf', type=float, default=0.1)         # 낮게 잡아 사람을 놓치지 않게 (추적이 잡음을 걸러 줌)
    parser.add_argument('--min-px', type=float, default=80)        # 분석 가능 기준: 상자 높이 80px (사용자 결정)
    parser.add_argument('--min-seconds', type=float, default=0.5)  # 후보가 되려면 분석 가능한 시간이 0.5초 이상
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    people = {}        # 추적 번호 → 그 사람의 집계(처음·마지막 프레임, 보인 프레임 수, 분석 가능 프레임 수, 가장 큰 모습)
    i = 0              # 지금 프레임 번호
    with (args.out_dir / 'tracks.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'track', 'x0', 'y0', 'x1', 'y1', 'conf'])
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            # persist=True: 앞 프레임의 추적 상태를 이어 받아 같은 사람에게 같은 번호를 준다. classes=[0]: 사람만
            boxes = model.track(bgr, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                tracker='bytetrack.yaml', verbose=False)[0].boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, track, conf in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
                    x0, y0, x1, y1 = xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height
                    writer.writerow([i, track, round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5), round(conf, 4)])
                    px = xyxy[3] - xyxy[1]                          # 상자 높이(픽셀)
                    inside = x0 > EDGE and y0 > EDGE and x1 < 1 - EDGE and y1 < 1 - EDGE
                    p = people.setdefault(track, dict(track=track, first=i, last=i, frames=0, big=0, best=-1.0))
                    p['last'], p['frames'] = i, p['frames'] + 1
                    p['big'] += px >= args.min_px and inside        # 분석 가능한 프레임 수 (True = 1로 더해짐)
                    if inside and px > p['best']:                   # 화면 안에 다 들어온 모습 중 가장 큰 것을 그림으로 남긴다
                        p['best'], p['best_frame'] = px, i
                        # 상자 긴 변 × 1.3 정사각형으로 잘라 160×160 그림으로 (화면 밖 부분은 잘려 나감)
                        side = max(xyxy[2] - xyxy[0], px) * 1.3
                        cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                        a, b = int(max(0, cy - side / 2)), int(max(0, cx - side / 2))
                        crop = bgr[a:int(cy + side / 2), b:int(cx + side / 2)]
                        # 줄일 때는 INTER_AREA(깔끔함), 키울 때는 INTER_CUBIC(덜 뭉개짐)
                        p['thumb'] = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA if side > 160 else cv2.INTER_CUBIC)
                        p['box'] = [round(v, 5) for v in (x0, y0, x1, y1)]
            i += 1
            if args.progress and i % 20 == 0:
                (args.out_dir / 'progress.json').write_text(json.dumps(dict(stage='people', done=i, total=total)))
    cap.release()

    # 후보 고르기: 가장 크게 보인 사람부터, 분석 가능한 시간이 충분하고 그림이 있는 사람만
    candidates = []
    for p in sorted(people.values(), key=lambda p: -p['best']):
        if p['big'] / fps < args.min_seconds or 'thumb' not in p:
            continue
        name = 'thumb_%d.jpg' % p['track']
        cv2.imwrite(str(args.out_dir / name), p['thumb'], [cv2.IMWRITE_JPEG_QUALITY, 88])
        candidates.append(dict(track=p['track'], thumb=name, best_frame=p['best_frame'], best_seconds=round(p['best_frame'] / fps, 2),
                               best_box=p['box'], max_px=round(p['best']), first_seconds=round(p['first'] / fps, 2),
                               last_seconds=round(p['last'] / fps, 2), analysable_seconds=round(p['big'] / fps, 2)))
    (args.out_dir / 'candidates.json').write_text(json.dumps(
        dict(video=dict(width=width, height=height, fps=fps, frames=i), people_seen=len(people), candidates=candidates),
        indent=1), encoding='utf-8')
    if args.progress:
        (args.out_dir / 'progress.json').write_text(json.dumps(dict(stage='people', done=i, total=i)))
    print('frames %d | people seen %d | candidates %d' % (i, len(people), len(candidates)))
    for c in candidates:
        print('  track %(track)d: largest %(max_px)dpx at %(best_seconds).1fs, analysable %(analysable_seconds).1fs' % c)


if __name__ == '__main__':
    main()

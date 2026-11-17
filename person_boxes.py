"""영상(한 편 또는 여러 편)의 모든 프레임에서 가장 확실한 사람 상자 하나를 찾는다 (YOLO, PyTorch).

TensorFlow와 PyTorch를 한 프로세스에 함께 올리지 않으므로 별도 환경(.yolo-venv)에서 돈다.
만든 CSV는 movenet_video_eval.py / movenet_video_check.py가 --boxes로 읽어 추적 시작점으로 쓰고,
ski24_eval.py가 '사람 검출 기준 잘라 넣기'에 쓴다.

출력 칸: video, split, frame, x0, y0, x1, y1(프레임 대비 비율), conf(확신도), people(그 프레임의 사람 수)
  EPFL 폴더 구조(<폴더>/<영상>/<시퀀스>.mp4)면 video·split을 채우고, 파일 한 편이면 비워 둔다.
  사람이 없던 프레임은 상자 칸이 빈 줄로 남는다.

실행 (프로젝트 루트에서)
  .yolo-venv\\Scripts\\python.exe person_boxes.py data/user_videos/clip.mp4 --out data/clip_boxes.csv
  .yolo-venv\\Scripts\\python.exe person_boxes.py data/ski2dpose_videos/Videos --out data/epfl_boxes.csv
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='one video file, or a folder of <video>/<split>.mp4')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', default='yolo11m.pt')
    parser.add_argument('--imgsz', type=int, default=1280)     # 멀리 있는 작은 스키어도 잡도록 큰 입력
    parser.add_argument('--conf', type=float, default=0.1)     # 낮은 확신도까지 받아 놓치지 않게
    args = parser.parse_args()

    # 입력이 파일 하나면 그것만, 폴더면 <영상>/<시퀀스>.mp4를 모두
    if args.source.is_file():
        videos = [('', '', args.source)]
    else:
        videos = [(p.parent.name, p.stem, p) for p in sorted(args.source.glob('*/*.mp4'))]
    model = YOLO(args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total, found, started = 0, 0, time.time()
    with args.out.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'split', 'frame', 'x0', 'y0', 'x1', 'y1', 'conf', 'people'])
        for number, (video, split, path) in enumerate(videos, 1):
            cap = cv2.VideoCapture(str(path))
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                h, w = bgr.shape[:2]
                boxes = model.predict(bgr, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)[0].boxes   # 사람(0)만
                if boxes is not None and len(boxes):
                    best = int(boxes.conf.argmax())          # 가장 확실한 한 명
                    x0, y0, x1, y1 = boxes.xyxy[best].tolist()
                    writer.writerow([video, split, i, round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5),
                                     round(y1 / h, 5), round(float(boxes.conf[best]), 4), len(boxes)])
                    found += 1
                else:
                    writer.writerow([video, split, i, '', '', '', '', '', 0])
                i += 1
                total += 1
            cap.release()
            if number % 10 == 0 or number == len(videos):   # 10편마다 진행 상황 출력
                print('%d/%d videos | frames %d | person found %.1f%% | %.1f ms/frame'
                      % (number, len(videos), total, 100 * found / max(1, total), 1000 * (time.time() - started) / max(1, total)),
                      flush=True)


if __name__ == '__main__':
    main()

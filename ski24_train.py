"""ski24_dataset.py가 만든 24점 스키어 그림으로 YOLO11m-Pose를 추가 학습한다.

바탕 가중치는 COCO(사람 몸 17점)로 학습된 yolo11m-pose.pt다. 점 개수가 17 → 24로 바뀌므로
관절을 내는 마지막 부분(머리 부분, head)은 새로 만들어지고, 나머지는 이미 배운 것을 이어서 조정한다.
설정값을 이 파일에 적어 두어 누가 돌려도 같은 조건으로 다시 학습할 수 있게 한다.

결과: data/ski24/runs/<이름>/weights/best.pt  (검증 영상에서 가장 좋았던 회차의 가중치)
      같은 폴더에 학습 곡선(results.csv, 그림)도 생긴다.

실행
  .yolo-venv\\Scripts\\python.exe ski24_train.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

SETTINGS = dict(
    imgsz=512,          # 입력 크기. 잘라 낸 그림 한 변의 중앙값이 약 530px이고, 스키·폴 끝은 가늘어서 더 줄이지 않는다
    epochs=150,         # 최대 학습 회차
    patience=30,        # 검증 영상 점수가 30회차 동안 나아지지 않으면 일찍 멈춘다
    batch=16,           # 한 번에 넣는 그림 수. GTX 1070 Ti(8GB)에 들어가는 크기
    degrees=10,         # 그림을 ±10°까지 돌려 학습한다. 스키어는 기울어진 자세가 많아서 현실적인 변형이다
    fliplr=0.5,         # 절반 확률로 좌우를 뒤집는다. 왼쪽·오른쪽 점은 ski24.yaml의 flip_idx로 맞바꾼다
    seed=0,             # 무작위 요소를 고정해 같은 결과가 나오게 한다
    workers=4,          # 그림을 읽어 오는 보조 프로세스 수
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=Path('data/ski24/ski24.yaml'))
    parser.add_argument('--weights', default='data/models/yolo11m-pose.pt')    # 바탕 가중치
    parser.add_argument('--name', default='yolo11m_pose_512')                   # 결과 폴더 이름
    args = parser.parse_args()

    model = YOLO(args.weights)
    # exist_ok=True: 같은 이름 폴더가 있으면 새 번호 폴더를 만들지 않고 그 자리에 쓴다
    model.train(data=str(args.data), project=str(Path('data/ski24/runs').resolve()), name=args.name,
                exist_ok=True, **SETTINGS)


if __name__ == '__main__':
    main()

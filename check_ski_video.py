"""Check the first ten seconds of an MP4 with MediaPipe Pose Landmarker."""
import argparse
import json
import math
from pathlib import Path

from ski_pose import knee_angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("pose_landmarker_lite.task"))
    args = parser.parse_args()
    for path in (args.video, args.model):
        if not path.is_file():
            parser.error(f"Missing file: {path}")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(args.video))
    rows = []
    try:
        if not cap.isOpened():
            parser.error("Video could not be opened")
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=args.model.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
        )
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not math.isfinite(timestamp) or timestamp < 0:
                    parser.error("Invalid frame timestamp")
                ms = round(timestamp)
                if rows and ms <= rows[-1]["time_ms"]:
                    parser.error("Decoder did not provide increasing timestamps")
                if ms >= 10000:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                poses = detector.detect_for_video(image, ms).pose_landmarks
                row = {"time_ms": ms, "pose_detected": bool(poses)}
                for side, ids in (("left", (23, 25, 27)), ("right", (24, 26, 28))):
                    flexion = None
                    if poses:
                        points = [poses[0][idx] for idx in ids]
                        leg = [(p.x * frame.shape[1], p.y * frame.shape[0],
                                min(p.visibility, p.presence) >= 0.5) for p in points]
                        angle = knee_angle(leg, 0)
                        if angle is not None:
                            flexion = round(180 - angle, 2)
                    row[side + "_flexion_deg"] = flexion
                rows.append(row)
    finally:
        cap.release()
    if not rows:
        parser.error("No frames decoded")
    print(json.dumps({"frames": len(rows), "detected_frames": sum(r["pose_detected"] for r in rows),
                      "samples": rows}, indent=2))


if __name__ == "__main__":
    main()

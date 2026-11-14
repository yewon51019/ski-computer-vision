"""Measure how well off-the-shelf pose models recover 2D knee flexion on EPFL Ski-2DPose.

For every image the script runs one or more pose methods, then compares the
predicted hip-knee-ankle angle of each leg with the angle computed from the
manual Ski-2DPose annotation.

Methods
  movenet_full      MoveNet SinglePose Thunder v4 on the whole frame (resize + pad to 256)
  yolo_det          (helper pass) YOLO person detector, most confident box; saved for movenet_detcrop
  movenet_detcrop   person box from the YOLO detector -> square crop -> MoveNet
  movenet_gtcrop    square crop around the annotated skeleton -> MoveNet (oracle upper bound)
  yolo_pose         Ultralytics YOLO-Pose on the whole frame, most confident person

Usage
  # pass 1 (PyTorch): YOLO-Pose + person boxes
  python eval/knee_eval.py --images DIR --labels LABELS --methods yolo_pose,yolo_det --tag yolo
  # pass 2 (TensorFlow): MoveNet three ways
  python eval/knee_eval.py --images DIR --labels LABELS --movenet SAVEDMODEL_DIR \
      --methods movenet_full,movenet_detcrop,movenet_gtcrop --boxes results/per_image_yolo.csv --tag movenet
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image

# Ski-2DPose joint order (24 joints)
SKI = dict(neck=1, hip_r=10, knee_r=11, ankle_r=12, hip_l=13, knee_l=14, ankle_l=15)
# COCO-17 order used by MoveNet and YOLO-Pose
COCO = dict(hip_l=11, hip_r=12, knee_l=13, knee_r=14, ankle_l=15, ankle_r=16)

KP_THRESH = 0.3  # keypoint confidence needed to count a joint as "measured"


def knee_angle(hip, knee, ankle):
    a, b = np.asarray(hip, float) - knee, np.asarray(ankle, float) - knee
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))  # 180 = straight leg


def load_labels(path, image_dir):
    d = json.load(open(path))
    recs = []
    for vid, splits in d.items():
        for split, frames in splits.items():
            for fid, e in frames.items():
                f = os.path.join(image_dir, vid, split, fid + ".webp")
                recs.append(dict(id=fid, path=f, ann=np.array(e["annotation"], float)))
    return recs


# ---------------------------------------------------------------- MoveNet
class MoveNet:
    def __init__(self, model_dir):
        import tensorflow as tf
        self.tf = tf
        self.fn = tf.saved_model.load(model_dir).signatures["serving_default"]

    def __call__(self, img):  # img: HxWx3 uint8 -> (17,2) xy px, (17,) score
        h, w = img.shape[:2]
        s = max(h, w)
        pad = np.zeros((s, s, 3), np.uint8)
        oy, ox = (s - h) // 2, (s - w) // 2
        pad[oy:oy + h, ox:ox + w] = img
        x = self.tf.image.resize(pad[None], (256, 256))
        out = self.fn(input=self.tf.cast(x, self.tf.int32))["output_0"].numpy()[0, 0]
        xy = np.stack([out[:, 1] * s - ox, out[:, 0] * s - oy], 1)
        return xy, out[:, 2]


def square_crop(img, box, margin=0.25):
    """box = x0,y0,x1,y1 px. Returns crop and its top-left offset (crop may extend past the frame, zero padded)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = max(side, 32)
    ax, ay = int(round(cx - side / 2)), int(round(cy - side / 2))
    S = int(round(side))
    H, W = img.shape[:2]
    out = np.zeros((S, S, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(ax, 0), max(ay, 0), min(ax + S, W), min(ay + S, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax] = img[sy0:sy1, sx0:sx1]
    return out, (ax, ay)


# ---------------------------------------------------------------- YOLO
class Yolo:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.m = YOLO(weights)

    def best(self, img_rgb, **kw):
        r = self.m.predict(img_rgb[..., ::-1], verbose=False, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        return r, i


def run(args):
    recs = load_labels(args.labels, args.images)
    if args.limit:
        recs = recs[::max(1, len(recs) // args.limit)][:args.limit]
    methods = args.methods.split(",")
    mn = MoveNet(args.movenet) if any(m.startswith("movenet") for m in methods) else None
    # TensorFlow and PyTorch clash (duplicate LLVM symbols) in one process, so the person
    # detector runs in a separate pass ("yolo_det") and MoveNet reads its boxes from CSV.
    det = Yolo(args.det_weights) if "yolo_det" in methods else None
    boxes = {}
    if "movenet_detcrop" in methods:
        import pandas as pd
        bdf = pd.read_csv(args.boxes)
        boxes = {r.id: r for r in bdf.itertuples()}
    pose = Yolo(args.pose_weights) if "yolo_pose" in methods else None

    rows = []
    t_tot = {m: 0.0 for m in methods}
    for n, r in enumerate(recs):
        img = np.asarray(Image.open(r["path"]).convert("RGB"))
        H, W = img.shape[:2]
        gt = r["ann"][:, :2] * [W, H]
        vis = r["ann"][:, 2]
        neck, midhip = gt[SKI["neck"]], (gt[SKI["hip_r"]] + gt[SKI["hip_l"]]) / 2
        row = dict(id=r["id"], W=W, H=H, torso_frac=float(np.linalg.norm(neck - midhip) / H),
                   torso_px=float(np.linalg.norm(neck - midhip)))
        for side in "rl":
            j = [SKI[f"hip_{side}"], SKI[f"knee_{side}"], SKI[f"ankle_{side}"]]
            row[f"gt_vis_{side}"] = int((vis[j] == 1).all())
            row[f"gt_ang_{side}"] = knee_angle(*gt[j])
            row[f"gt_knee_{side}_x"], row[f"gt_knee_{side}_y"] = gt[j[1]]

        for m in methods:
            t0 = time.time()
            xy = sc = None
            if m == "movenet_full":
                xy, sc = mn(img)
            elif m == "movenet_gtcrop":
                crop, (ax, ay) = square_crop(img, (*gt.min(0), *gt.max(0)), margin=0.15)
                xy, sc = mn(crop)
                xy = xy + [ax, ay]
            elif m == "yolo_det":
                b = det.best(img, classes=[0], conf=0.1, imgsz=args.det_imgsz)
                row["det_conf"] = np.nan
                if b is not None:
                    res, i = b
                    row["det_x0"], row["det_y0"], row["det_x1"], row["det_y1"] = res.boxes.xyxy[i].cpu().numpy()
                    row["det_conf"] = float(res.boxes.conf[i])
                t_tot[m] += time.time() - t0
                continue
            elif m == "movenet_detcrop":
                b = boxes[r["id"]]
                if not np.isnan(b.det_conf):
                    crop, (ax, ay) = square_crop(img, (b.det_x0, b.det_y0, b.det_x1, b.det_y1), margin=0.15)
                    xy, sc = mn(crop)
                    xy = xy + [ax, ay]
            elif m == "yolo_pose":
                b = pose.best(img, conf=0.1, imgsz=args.pose_imgsz)
                if b is not None:
                    res, i = b
                    xy = res.keypoints.xy[i].cpu().numpy()
                    sc = res.keypoints.conf[i].cpu().numpy()
            t_tot[m] += time.time() - t0
            for side, other in (("r", "l"), ("l", "r")):
                j = [COCO[f"hip_{side}"], COCO[f"knee_{side}"], COCO[f"ankle_{side}"]]
                if xy is None:
                    row[f"{m}_ok_{side}"] = 0
                    row[f"{m}_ang_{side}"] = np.nan
                    row[f"{m}_knee_{side}_x"] = row[f"{m}_knee_{side}_y"] = np.nan
                    continue
                row[f"{m}_ok_{side}"] = int((sc[j] >= KP_THRESH).all())
                row[f"{m}_ang_{side}"] = knee_angle(*xy[j])
                row[f"{m}_knee_{side}_x"], row[f"{m}_knee_{side}_y"] = xy[j[1]]
        rows.append(row)
        if n % 100 == 0:
            print(n, len(recs), {k: round(v / (n + 1), 3) for k, v in t_tot.items()}, flush=True)

    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"per_image_{args.tag}.csv"), index=False)
    json.dump({k: v / len(recs) for k, v in t_tot.items()},
              open(os.path.join(args.out, f"sec_per_image_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--movenet")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="run")
    p.add_argument("--methods", default="movenet_full,movenet_gtcrop")
    p.add_argument("--boxes", help="CSV from a run with --methods yolo_det (needed by movenet_detcrop)")
    p.add_argument("--det_weights", default="yolo11m.pt")
    p.add_argument("--pose_weights", default="yolo11m-pose.pt")
    p.add_argument("--det_imgsz", type=int, default=1280)
    p.add_argument("--pose_imgsz", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0)
    run(p.parse_args())

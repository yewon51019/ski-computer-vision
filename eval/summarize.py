"""Summarize per-image CSVs from knee_eval.py into per-size-group metrics.

Evaluated images: both legs have hip, knee and ankle annotated as visible (1,527 of 1,982).
Size groups: torso length (neck to mid-hip) / image height, split at the median over all 1,982 images.

Metrics (per method x group)
  both_knees_rate   share of evaluated images where hip, knee, ankle of BOTH legs have score >= 0.3
  knee_hit_rate     of those, share where both predicted knees land within 0.5 torso lengths of the
                    annotated knees (i.e. the model found the right skier and the right joints)
  mae_deg           mean absolute knee angle error over both legs, among measured images
  median_deg        median of the per-leg absolute error
  within10 / within20   share of legs with error <= 10 / 20 degrees
Left and right are matched by whichever assignment gives the lower error (models often swap
sides on back views); the strict L=L, R=R error is reported as mae_strict_deg.
"""
import argparse
import json

import numpy as np
import pandas as pd

METHODS = ["movenet_full", "movenet_detcrop", "movenet_gtcrop", "yolo_pose"]


def per_image(df, m):
    g = df[["gt_ang_r", "gt_ang_l"]].to_numpy()
    p = df[[f"{m}_ang_r", f"{m}_ang_l"]].to_numpy()
    direct = np.abs(p - g)
    swap = np.abs(p[:, ::-1] - g)
    use_swap = swap.sum(1) < direct.sum(1)
    err = np.where(use_swap[:, None], swap, direct)
    gk = df[["gt_knee_r_x", "gt_knee_r_y", "gt_knee_l_x", "gt_knee_l_y"]].to_numpy().reshape(-1, 2, 2)
    pk = df[[f"{m}_knee_r_x", f"{m}_knee_r_y", f"{m}_knee_l_x", f"{m}_knee_l_y"]].to_numpy().reshape(-1, 2, 2)
    pk_sw = np.where(use_swap[:, None, None], pk[:, ::-1], pk)
    dist = np.linalg.norm(pk_sw - gk, axis=2) / df["torso_px"].to_numpy()[:, None]
    ok = (df[f"{m}_ok_r"] == 1) & (df[f"{m}_ok_l"] == 1)
    return ok.to_numpy(), err, direct, (dist <= 0.5).all(1)


def summarize(df, m):
    ok, err, direct, hit = per_image(df, m)
    n = len(df)
    e = err[ok].ravel()
    return dict(n_eval=n, n_measured=int(ok.sum()),
                both_knees_rate=ok.mean(),
                knee_hit_rate=hit[ok].mean() if ok.any() else np.nan,
                mae_deg=e.mean() if len(e) else np.nan,
                median_deg=np.median(e) if len(e) else np.nan,
                within10=(e <= 10).mean() if len(e) else np.nan,
                within20=(e <= 20).mean() if len(e) else np.nan,
                mae_strict_deg=direct[ok].mean() if ok.any() else np.nan)


def main(a):
    dfs = [pd.read_csv(f) for f in a.csv]
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d[["id"] + [c for c in d.columns if c not in df.columns]], on="id")
    thr = df["torso_frac"].median()
    df["group"] = np.where(df["torso_frac"] < thr, "small", "large")
    ev = df[(df.gt_vis_r == 1) & (df.gt_vis_l == 1)]
    out = []
    for m in [m for m in METHODS if f"{m}_ok_r" in df]:
        for gname, gdf in [("all", ev), ("small", ev[ev.group == "small"]), ("large", ev[ev.group == "large"])]:
            out.append(dict(method=m, group=gname, **summarize(gdf, m)))
    res = pd.DataFrame(out)
    # Common subset: images every method measured, so angle errors are compared on the same photos
    ms = [m for m in METHODS if f"{m}_ok_r" in df]
    common = np.ones(len(ev), bool)
    for m in ms:
        common &= per_image(ev, m)[0]
    cs = []
    for m in ms:
        for gname in ["all", "small", "large"]:
            sel = common & ((ev.group == gname).to_numpy() if gname != "all" else True)
            e = per_image(ev[sel], m)[1].ravel()
            cs.append(dict(method=m, group=gname, n_common=int(sel.sum()), mae_deg=e.mean(),
                           median_deg=np.median(e), within10=(e <= 10).mean()))
    common_df = pd.DataFrame(cs)
    pd.set_option("display.width", 200)
    print(f"torso/height median threshold = {thr:.4f}; evaluated images = {len(ev)} "
          f"(small {int((ev.group == 'small').sum())}, large {int((ev.group == 'large').sum())})")
    print(res.round(3).to_string(index=False))
    print("\nSame-photo comparison (images measured by every method):")
    print(common_df.round(3).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
        common_df.to_csv(a.out.replace(".csv", "_common.csv"), index=False)
        df.to_csv(a.out.replace(".csv", "_merged_per_image.csv"), index=False)
        json.dump(dict(torso_frac_median=thr, n_eval=len(ev)), open(a.out.replace(".csv", "_meta.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--out")
    main(p.parse_args())

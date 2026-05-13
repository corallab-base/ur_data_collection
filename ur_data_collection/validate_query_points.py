#!/usr/bin/env python3
"""Validate SAM2 query points across episodes before post-processing.

Loads episodes from one or more directories in order and displays frame 0 of
each with the query point that SamuraiFoundationPoseProcessor would use
overlaid as a crosshair.  This lets you verify the point is on the object
before running the full (slow) pose estimation pipeline.

Query-point priority (mirrors post_processor.py):
  1. Projected 2D centre of the last valid pose from the previous episode
     (requires that episode to already have 'obj_pose_4x4').
  2. Fixed default — (0.5, 0.5) or --query-point x y.

Usage
-----
    validate_query_points <dir1> [dir2 …] [--query-point 0.5 0.5] [--display-dim 480]

Navigation
----------
    Space / Enter / → : next episode
    b / ←             : previous episode
    q / Esc           : quit
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np


def _project_pose_to_frac(
    pose: np.ndarray,
    K: np.ndarray,
    img_shape: tuple,
) -> tuple:
    """Project a 4×4 camera-frame pose origin to (x_frac, y_frac) in [0, 1]²."""
    t = pose[:3, 3]
    if t[2] <= 0:
        return (0.5, 0.5)
    u = K[0, 0] * t[0] / t[2] + K[0, 2]
    v = K[1, 1] * t[1] / t[2] + K[1, 2]
    H, W = img_shape[:2]
    return (float(np.clip(u / W, 0.0, 1.0)), float(np.clip(v / H, 0.0, 1.0)))


def _collect_episode_files(folders: list[str]) -> list[str]:
    """Return sorted episode pkl paths from each folder, preserving folder order."""
    files = []
    for folder in folders:
        found = sorted(glob.glob(os.path.join(folder, "episode_*.pkl")))
        if not found:
            print(f"  Warning: no episode_*.pkl in {folder!r}")
        files.extend(found)
    return files


def _compute_query_points(
    episode_files: list[str],
    fixed_qp: tuple | None,
) -> tuple[list[tuple], list[str]]:
    """Simulate the per-episode query-point selection.

    If fixed_qp is given, every episode uses it and pose propagation is
    skipped (mirrors the post_processor behaviour when query_point is set).
    Otherwise each episode gets the projected last pose from the previous
    episode when available, falling back to (0.5, 0.5).

    Returns (query_points, sources) where source is one of:
      "fixed", "prev pose", "default".
    """
    qps = []
    sources = []
    last_qp: tuple | None = None

    for path in episode_files:
        if fixed_qp is not None:
            qps.append(fixed_qp)
            sources.append("fixed")
            continue

        if last_qp is not None:
            qps.append(last_qp)
            sources.append("prev pose")
        else:
            qps.append((0.5, 0.5))
            sources.append("default")

        try:
            with open(path, "rb") as f:
                ep = pickle.load(f)
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
            continue

        poses = ep.get("obj_pose_4x4", [])
        K = ep.get("camera_K")
        imgs = ep.get("img", [])
        if not poses or K is None or not imgs:
            continue

        last_pose = next((p for p in reversed(poses) if p is not None), None)
        if last_pose is not None:
            last_qp = _project_pose_to_frac(last_pose, K, imgs[0].shape)

    return qps, sources


def _load_frame0(path: str) -> np.ndarray | None:
    """Load only the first image from an episode pickle."""
    try:
        with open(path, "rb") as f:
            ep = pickle.load(f)
        frames = ep.get("img", [])
        return frames[0] if frames else None
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def _has_poses(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            ep = pickle.load(f)
        poses = ep.get("obj_pose_4x4", [])
        return any(p is not None for p in poses)
    except Exception:
        return False


def _draw_query_point(
    img: np.ndarray,
    qp: tuple,
    display_dim: int,
    label_top: str,
    label_bot: str,
    has_poses: bool,
) -> np.ndarray:
    vis = cv2.resize(img, (display_dim, display_dim), interpolation=cv2.INTER_AREA)

    px = int(qp[0] * display_dim)
    py = int(qp[1] * display_dim)

    cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(vis, (px, py), 10, (0, 0, 255), 2, cv2.LINE_AA)

    # Processed indicator
    if has_poses:
        cv2.circle(vis, (display_dim - 14, 14), 7, (0, 220, 0), -1)

    # Labels
    cv2.putText(vis, label_top, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label_top, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, label_bot, (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label_bot, (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Validate SAM2 query points across episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs", nargs="+", metavar="DIR",
        help="One or more directories containing episode_*.pkl files",
    )
    parser.add_argument(
        "--query-point", nargs=2, type=float, default=None,
        metavar=("X", "Y"),
        help="Fix the SAM2 query point for all episodes (image fractions). "
             "If omitted, each episode uses the projected pose from the previous one.",
    )
    parser.add_argument(
        "--display-dim", type=int, default=480, metavar="PX",
        help="Display window size in pixels (default: 480)",
    )
    args = parser.parse_args()

    fixed_qp = tuple(args.query_point) if args.query_point is not None else None

    episode_files = _collect_episode_files(args.dirs)
    if not episode_files:
        print("No episodes found.")
        sys.exit(1)

    print(f"Found {len(episode_files)} episode(s). Pre-computing query points …")
    query_points, sources = _compute_query_points(episode_files, fixed_qp)
    is_processed = [_has_poses(p) for p in episode_files]
    print("Done. Use Space/Enter/→ to advance, b/← to go back, q/Esc to quit.\n")

    idx = 0
    cv2.namedWindow("Query Point Validation", cv2.WINDOW_AUTOSIZE)

    while True:
        path = episode_files[idx]
        qp = query_points[idx]
        source = sources[idx]
        processed = is_processed[idx]

        frame = _load_frame0(path)
        if frame is None:
            print(f"  [{idx + 1}/{len(episode_files)}] No image in {os.path.basename(path)}, skipping.")
            idx = min(idx + 1, len(episode_files) - 1)
            continue

        label_top = f"{idx + 1}/{len(episode_files)}  {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        label_bot = f"qp=({qp[0]:.3f}, {qp[1]:.3f})  [{source}]{'  ✓ poses' if processed else ''}"

        vis = _draw_query_point(
            frame, qp, args.display_dim,
            label_top, label_bot, processed,
        )
        cv2.imshow("Query Point Validation", vis)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):        # q / Esc — quit
            break
        elif key in (ord(" "), 13, 83):  # Space / Enter / → — next
            idx = min(idx + 1, len(episode_files) - 1)
        elif key in (ord("b"), 81):      # b / ← — back
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline bulk post-processing for recorded episodes.

Usage
-----
    process_episodes <episode_dir> [options]

Each episode_*.pkl in <episode_dir> is processed in alphabetical order.
Episodes that lack a saved camera→world transform are skipped with a warning.
Processed episodes are saved back to the same .pkl files.

Post-processors are enabled by flag:
    --samurai-checkpoint  PATH   Run SAMURAI mask segmentation
    --mesh-path           PATH   Run FoundationPose 6D pose estimation
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Visualisation helpers (no ROS dependency)
# ---------------------------------------------------------------------------

def _draw_pose_axes(
    img: np.ndarray,
    pose: np.ndarray,
    K_recorded: np.ndarray,
    recorded_dim: int,
    display_dim: int,
    axis_len: float = 0.05,
) -> None:
    s = display_dim / recorded_dim
    K = K_recorded.copy() * s
    K[2, 2] = 1.0
    origin = pose[:3, 3]
    tips = origin + axis_len * pose[:3, :3].T

    def proj(pt):
        p = K @ pt
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    o = proj(origin)
    colors = [(0, 0, 220), (0, 220, 0), (220, 0, 0)]  # X=red, Y=green, Z=blue (BGR)
    for tip, color in zip(tips, colors):
        cv2.line(img, o, proj(tip), color, 2, cv2.LINE_AA)
    cv2.circle(img, o, 3, (255, 255, 255), -1)


def _interactive_preview(
    episode: dict,
    rate_hz: float = 15.0,
    display_dim: int = 480,
) -> None:
    frames = episode.get("img", [])
    N = len(frames)
    if not N:
        print("  [preview] No frames in episode.")
        return

    masks = episode.get("mask", [])
    poses = episode.get("obj_pose_4x4", [])
    K = episode.get("camera_K")
    target_dim = frames[0].shape[0]
    frame_ms = max(1, int(1000.0 / rate_hz))
    idx = 0

    print("  [preview] Q / ESC to continue to next episode.")
    while True:
        frame = frames[idx]
        display = cv2.resize(frame, (display_dim, display_dim))

        if idx < len(masks):
            m = cv2.resize(
                masks[idx], (display_dim, display_dim),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = display.copy()
            overlay[m > 0] = (0, 200, 0)
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        if K is not None and idx < len(poses):
            _draw_pose_axes(display, poses[idx], K, target_dim, display_dim)

        cv2.putText(
            display, f"{idx + 1}/{N}", (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.imshow("Episode Preview", display)
        key = cv2.waitKey(frame_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        idx = (idx + 1) % N

    cv2.destroyWindow("Episode Preview")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline bulk post-processing for recorded episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "episode_dir_or_path",
        help="Directory containing or path to episode_*.pkl files",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Preview each episode interactively after processing",
    )
    parser.add_argument(
        "--samurai-checkpoint", default="", metavar="PATH",
        help="SAM2 checkpoint (.pt) — enables SAMURAI mask segmentation",
    )
    parser.add_argument(
        "--mesh-path", default="", metavar="PATH",
        help="Object mesh file — enables FoundationPose 6D pose estimation",
    )
    parser.add_argument(
        "--apply-scale", type=float, default=1.0, metavar="FLOAT",
        help="Uniform scale applied to mesh vertices before FoundationPose (default: 1.0)",
    )
    parser.add_argument(
        "--display-dim", type=int, default=480, metavar="PX",
        help="Preview window size in pixels (default: 480)",
    )
    args = parser.parse_args()

    # --- Build post-processors ---
    post_processors = []

    if args.samurai_checkpoint:
        from ur_data_collection.post_processor import SamuraiPostProcessor
        print(f"Loading SAMURAI from {args.samurai_checkpoint} …")
        post_processors.append(SamuraiPostProcessor(checkpoint=args.samurai_checkpoint))
        print("SAMURAI ready.")

    if args.mesh_path:
        from ur_data_collection.post_processor import SamuraiFoundationPoseProcessor
        print(f"Loading FoundationPose for mesh {args.mesh_path} …")
        post_processors.append(
            SamuraiFoundationPoseProcessor(
                mesh_path=args.mesh_path,
                apply_scale=args.apply_scale,
            )
        )
        print("FoundationPose ready.")

    if not post_processors:
        print("No post-processors configured — pass --samurai-checkpoint and/or --mesh-path.")
        sys.exit(0)

    # --- Discover episodes ---
    if os.path.isdir(args.episode_dir_or_path):
        pattern = os.path.join(args.episode_dir_or_path, "episode_*.pkl")
        episode_files = sorted(glob.glob(pattern))
        if not episode_files:
            print(f"No episode_*.pkl files found in {args.episode_dir_or_path!r}")
            sys.exit(1)
    else:
        episode_files = [args.episode_dir_or_path]

    print(f"\nFound {len(episode_files)} episode(s) in {args.episode_dir_or_path!r}\n")

    n_processed = 0
    n_skipped = 0

    for ep_path in episode_files:
        ep_name = os.path.basename(ep_path)
        print(f"--- {ep_name} ---")

        with open(ep_path, "rb") as f:
            episode = pickle.load(f)

        # Require camera→world transform
        T_world_camera = episode.get("T_world_camera")
        if T_world_camera is None:
            print(
                f"  SKIPPED: 'T_world_camera' not found in episode.\n"
                "  Re-record with a collector version that saves the camera TF.\n"
            )
            n_skipped += 1
            continue

        # --- Deduplicate near-identical frames ---
        from ur_data_collection.post_processor import deduplicate_episode
        before = len(episode.get("img", []))
        deduplicate_episode(episode)
        after = len(episode.get("img", []))
        if after < before:
            print(f"  Deduplicated: {before} → {after} frames")

        # --- prepare() — GUI steps on the main thread ---
        for proc in post_processors:
            proc.prepare(episode)

        # --- process() ---
        try:
            for proc in post_processors:
                print(f"  Running {proc.__class__.__name__} …")
                additions = proc.process(episode)
                episode.update(additions)
                print(f"  {proc.__class__.__name__} done — keys added: {list(additions.keys())}")
        except Exception as e:
            import traceback
            print(f"  ERROR during processing: {e}")
            traceback.print_exc()
            n_skipped += 1
            continue

        # --- Transform object poses to world frame ---
        if "obj_pose_4x4" in episode:
            episode["obj_pose_4x4_world"] = [
                (T_world_camera @ p).astype(np.float64)
                for p in episode["obj_pose_4x4"]
            ]
            print(
                f"  Object poses transformed to world frame "
                f"({len(episode['obj_pose_4x4_world'])} poses → 'obj_pose_4x4_world')"
            )

        # --- Interactive preview ---
        if args.interactive:
            _interactive_preview(episode, display_dim=args.display_dim)

        # --- Save back ---
        with open(ep_path, "wb") as f:
            pickle.dump(episode, f)
        print(f"  Saved → {ep_path}\n")
        n_processed += 1

    print(f"Done. {n_processed} processed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()

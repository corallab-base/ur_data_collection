#!/usr/bin/env python3
"""Post-processing pipeline for annotating recorded episodes with object poses.

Usage:
    processor = SamuraiFoundationPoseProcessor(mesh_path="/path/to/obj.obj")
    additions = processor.process(episode)   # deduplicates episode in-place first
    episode.update(additions)
    # episode now has "obj_pose_4x4"

Episodes must already contain "mask" (recorded while SAMURAI was running),
"depth", and "camera_K". Near-duplicate frames are removed before annotation.
FoundationPose re-registers at each local blob-size peak to limit tracking drift.

The SAMURAI ROS node must be running during collection, configured to subscribe
to the live camera topic and publish to /{obj_name}/mask.

    ros2 run coral_trackers samurai_tracker --ros-args \
        -p rgb_topic:=/camera/camera/color/image_raw \
        -p object_name:=obj \
        -p samurai_checkpoint:=<path>
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm, trange

FP_ROS_PATH_DEFAULT = (
    "/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2"
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PostProcessor(ABC):
    @abstractmethod
    def process(self, episode: dict) -> dict:
        """Annotate an episode. Returns a dict of new keys to merge in."""


# ---------------------------------------------------------------------------
# SAMURAI + FoundationPose implementation
# ---------------------------------------------------------------------------

class SamuraiFoundationPoseProcessor(PostProcessor):
    """
    Uses per-frame masks already recorded in an episode (from a running SAMURAI
    node) to estimate 6D object pose via FoundationPose.

    Near-duplicate frames are removed in-place before processing. FoundationPose
    re-registers at each local blob-size peak (SAMURAI confidence peak) and
    tracks forward to the next anchor, keeping drift short. The first anchor
    also tracks backward to cover frame 0.

    Adds to the episode:
        obj_pose_4x4 : list[np.ndarray (4,4)]  — object-in-camera SE(3)
    """

    def __init__(
        self,
        mesh_path: str,
        apply_scale: float = 1.0,
        fp_ros_path: str = FP_ROS_PATH_DEFAULT,
    ):
        if fp_ros_path not in sys.path:
            sys.path.insert(0, fp_ros_path)
        fp_path = os.path.join(fp_ros_path, "FoundationPose")
        if fp_path not in sys.path:
            sys.path.insert(0, fp_path)

        from FoundationPose.estimater import (
            dr,
            FoundationPose,
            ScorePredictor,
            PoseRefinePredictor,
        )

        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(apply_scale)
        _, extents = trimesh.bounds.oriented_bounds(mesh)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self._est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
        )

    def process(self, episode: dict) -> dict:
        deduplicate_episode(episode)

        frames = episode["img"]    # list of (D,D,3) uint8 BGR
        depths = episode["depth"]  # list of (D,D) float32 metres
        masks = episode["mask"]    # list of (D,D) uint8
        K = episode.get("camera_K")
        if K is None:
            raise ValueError("Episode has no 'camera_K' — was it recorded with depth?")
        N = len(frames)
        if N == 0:
            return {"obj_pose_4x4": []}

        blob_sizes = [int((m > 0).sum()) for m in masks]
        if max(blob_sizes) == 0:
            raise ValueError(
                "All masks are empty — was SAMURAI running and tracking an object?"
            )

        anchors = _find_registration_anchors(blob_sizes)
        poses: list = [None] * N

        # # use the first registration anchor:
        # anchor = anchors[0]

        # rgb = cv2.cvtColor(frames[anchor], cv2.COLOR_BGR2RGB)
        # poses[anchor] = self._est.register(
        #     K=K, rgb=rgb, depth=depths[anchor],
        #     ob_mask=masks[anchor] > 0, iteration=10,
        # )
        # pose_at_anchor = self._est.pose_last.clone()

        # # Forward: anchor+1 to end of episode
        # for i in trange(
        #     anchor + 1, N,
        #     desc=f"  fwd [{anchor}→{N}]", leave=False, unit="frame",
        # ):
        #     rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        #     poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=10)

        # self._est.pose_last = pose_at_anchor.clone()
        # for i in trange(
        #         anchor - 1, -1, -1,
        #         desc=f"  bwd [{anchor}→0]", leave=False, unit="frame",
        # ):
        #     rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        #     poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=10)

        for seg_idx, anchor in enumerate(tqdm(anchors, desc="FP register", unit="anchor")):
            rgb = cv2.cvtColor(frames[anchor], cv2.COLOR_BGR2RGB)
            poses[anchor] = self._est.register(
                K=K, rgb=rgb, depth=depths[anchor],
                ob_mask=masks[anchor] > 0, iteration=8,
            )
            pose_at_anchor = self._est.pose_last.clone()

            # Forward: anchor+1 → next anchor (exclusive) or end of episode
            next_anchor = anchors[seg_idx + 1] if seg_idx + 1 < len(anchors) else N
            for i in trange(
                anchor + 1, next_anchor,
                desc=f"  fwd [{anchor}→{next_anchor - 1}]", leave=False, unit="frame",
            ):
                rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=8)

            # Backward from the first anchor to frame 0
            if seg_idx == 0 and anchor > 0:
                self._est.pose_last = pose_at_anchor.clone()
                for i in trange(
                    anchor - 1, -1, -1,
                    desc=f"  bwd [{anchor}→0]", leave=False, unit="frame",
                ):
                    rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                    poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=8)

        _visualize_anchors(frames, masks, poses, [anchor], K)
        return {"obj_pose_4x4": poses}

    def _anchor_translation(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """Shift pose_last translation to the mask's 3D centroid when available."""
        if self._est.pose_last is None:
            return
        center = _mean_xyz_from_mask(mask, depth, K)
        if center is None:
            return
        pl = self._est.pose_last
        p = pl.detach().cpu().numpy().reshape(4, 4).copy()
        p[:3, 3] = center
        self._est.pose_last = torch.from_numpy(p).reshape(pl.shape).to(pl.device)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def deduplicate_episode(
    episode: dict,
    min_pixel_diff: float = 2.0,
    max_gap: int = 5,
) -> None:
    """Remove near-duplicate frames from all list fields of the episode in-place.

    A frame is kept if either:
    - Its mean absolute pixel difference from the last kept frame >= min_pixel_diff, or
    - At least max_gap frames have elapsed since the last kept frame.
    """
    imgs = episode.get("img", [])
    N = len(imgs)
    if N < 2:
        return

    keep = [0]
    ref = imgs[0].astype(np.float32)
    for i in range(1, N):
        curr = imgs[i].astype(np.float32)
        diff = float(np.mean(np.abs(curr - ref)))
        if diff >= min_pixel_diff or (i - keep[-1]) >= max_gap:
            keep.append(i)
            ref = curr

    for key, val in list(episode.items()):
        if isinstance(val, list) and len(val) == N:
            episode[key] = [val[j] for j in keep]


def smooth_poses(
    poses: list,
    alpha: float = 0.3,
    trans_threshold: float = 0.05,
    rot_threshold: float = 0.5,
    gate_limit: int = 5,
) -> list:
    """EMA smoothing with outlier rejection on a list of 4×4 SE(3) matrices.

    Mirrors the pattern in coral_trackers/colors_tracker.py:
    - Translation: standard EMA, gated by Euclidean distance jump.
    - Rotation: quaternion EMA (sign-corrected to nearest hemisphere), gated by
      angular distance jump.
    - After gate_limit consecutive rejections the update is accepted regardless,
      so the filter can recover from genuine large displacements.

    Args:
        alpha: smoothing weight on the new measurement (1.0 = no smoothing).
        trans_threshold: max translation jump (m) before a frame is treated as outlier.
        rot_threshold: max rotation jump (rad) before a frame is treated as outlier.
        gate_limit: consecutive rejections allowed before forcing an accept.
    """
    result = [None] * len(poses)
    f_t: Optional[np.ndarray] = None   # filtered translation
    f_q: Optional[np.ndarray] = None   # filtered quaternion [w, x, y, z]
    gate = 0

    for i, pose in enumerate(poses):
        if pose is None:
            result[i] = None
            continue

        p = np.asarray(pose, dtype=np.float64)
        n_t = p[:3, 3].copy()
        n_q = _mat_to_quat(p[:3, :3])

        if f_t is None:
            f_t, f_q = n_t.copy(), n_q.copy()
            result[i] = p.copy()
            continue

        # Flip to nearest hemisphere so EMA stays meaningful
        if np.dot(n_q, f_q) < 0:
            n_q = -n_q

        trans_jump = float(np.linalg.norm(n_t - f_t))
        rot_jump = 2.0 * np.arccos(np.clip(abs(np.dot(n_q, f_q)), 0.0, 1.0))

        if (trans_jump < trans_threshold and rot_jump < rot_threshold) or gate >= gate_limit:
            f_t = alpha * n_t + (1.0 - alpha) * f_t
            f_q = alpha * n_q + (1.0 - alpha) * f_q
            f_q /= np.linalg.norm(f_q)
            gate = 0
        else:
            gate += 1

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = _quat_to_mat(f_q)
        out[:3, 3] = f_t
        result[i] = out.astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _visualize_anchors(
    frames: list,
    masks: list,
    poses: list,
    anchors: list,
    K: np.ndarray,
    out_dir: str = "/tmp",
    axis_len: float = 0.05,
    display_dim: int = 480,
) -> None:
    """Save one PNG per anchor frame showing the mask overlay and pose axes."""
    for idx, anchor in enumerate(anchors):
        pose = poses[anchor]
        if pose is None:
            continue

        frame = frames[anchor]
        vis = cv2.resize(frame, (display_dim, display_dim), interpolation=cv2.INTER_AREA)

        # Mask overlay (green, semi-transparent)
        if masks and anchor < len(masks):
            m = cv2.resize(
                masks[anchor], (display_dim, display_dim),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = vis.copy()
            overlay[m > 0] = (0, 200, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # Project and draw XYZ axes
        s = display_dim / frame.shape[0]
        Kd = K.copy() * s
        Kd[2, 2] = 1.0
        origin = pose[:3, 3]
        tips = origin + axis_len * pose[:3, :3].T  # rows: X, Y, Z tips

        def proj(pt):
            p = Kd @ pt
            return (int(p[0] / p[2]), int(p[1] / p[2]))

        o = proj(origin)
        for tip, color in zip(tips, [(0, 0, 220), (0, 220, 0), (220, 0, 0)]):
            cv2.line(vis, o, proj(tip), color, 2, cv2.LINE_AA)
        cv2.circle(vis, o, 4, (255, 255, 255), -1)

        # Translation label
        t = pose[:3, 3]
        cv2.putText(
            vis,
            f"anchor {idx}  frame {anchor}  t=[{t[0]:.3f} {t[1]:.3f} {t[2]:.3f}]",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        path = os.path.join(out_dir, f"fp_anchor_{idx:02d}_frame_{anchor:04d}.png")
        cv2.imwrite(path, vis)
        print(f"  anchor {idx}: frame {anchor}  → {path}")


def _find_registration_anchors(
    blob_sizes: list,
    min_separation: int = 15,
    min_blob_fraction: float = 0.4,
) -> list:
    """Return indices of local blob-size maxima to use as FP registration anchors.

    Peaks must exceed min_blob_fraction * global_max. Within min_separation
    frames of each other, only the taller peak is kept.
    """
    arr = np.array(blob_sizes, dtype=float)
    N = len(arr)
    threshold = min_blob_fraction * arr.max()

    candidates = []
    for i in range(N):
        if arr[i] < threshold:
            continue
        left = arr[i - 1] if i > 0 else -np.inf
        right = arr[i + 1] if i < N - 1 else -np.inf
        if arr[i] >= left and arr[i] >= right:
            candidates.append(i)

    # Non-maximum suppression: within min_separation, keep the taller peak
    peaks = []
    for c in candidates:
        if peaks and c - peaks[-1] < min_separation:
            if arr[c] > arr[peaks[-1]]:
                peaks[-1] = c
        else:
            peaks.append(c)

    if not peaks:
        peaks = [int(np.argmax(arr))]

    return peaks


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → [w, x, y, z] unit quaternion (Shepperd's method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] unit quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _mean_xyz_from_mask(
    mask_u8: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    mad_threshold: float = 2.5,
) -> Optional[np.ndarray]:
    """Mean 3D point of mask pixels, with MAD-based depth outlier rejection.

    Adapted from coral_trackers/mask_center_tracker.py.
    K must be a 3×3 camera matrix.
    Returns (3,) float32 [X, Y, Z] in camera frame, or None if insufficient depth.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    zs = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0)
    if not np.any(valid):
        return None
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]

    median_z = float(np.median(zs))
    mad = float(np.median(np.abs(zs - median_z)))
    if mad > 0:
        inliers = np.abs(zs - median_z) <= mad_threshold * mad
        xs, ys, zs = xs[inliers], ys[inliers], zs[inliers]
    if zs.size == 0:
        return None

    return np.array(
        [
            (xs - cx) @ zs / (fx * zs.size),
            (ys - cy) @ zs / (fy * zs.size),
            zs.mean(),
        ],
        dtype=np.float32,
    )

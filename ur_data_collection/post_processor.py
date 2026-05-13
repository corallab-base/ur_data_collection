#!/usr/bin/env python3
"""Post-processing pipeline for annotating recorded episodes with object masks and poses.

Typical offline workflow
------------------------
    samurai = SamuraiPostProcessor(checkpoint="/path/to/sam2.pt")
    fp      = SamuraiFoundationPoseProcessor(mesh_path="/path/to/obj.obj")

    # bbox [x1,y1,x2,y2] in the episode's target_img_dim pixel space,
    # drawn by the user on frame 0 via cv2.selectROI before this call.
    episode.update(samurai.process(episode, bbox))   # adds "mask"
    episode.update(fp.process(episode))              # adds "obj_pose_4x4"
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

SAMURAI_PATH_DEFAULT = os.path.expanduser("~/phd/software/samurai/sam2")
SAMURAI_CONFIG_DEFAULT = "configs/samurai/sam2.1_hiera_b+.yaml"
SAMURAI_CHECKPOINT_DEFAULT = os.path.expanduser(
    "~/phd/software/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)

FP_ROS_PATH_DEFAULT = (
    "/home/tassos/phd/research/demos/goc_demo_workspace/src/FoundationPoseROS2"
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PostProcessor(ABC):
    def prepare(self, episode: dict) -> None:
        """Called on the main thread before process(). Override for GUI steps."""

    @abstractmethod
    def process(self, episode: dict) -> dict:
        """Annotate an episode in a background thread. Returns new keys to merge."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_roi_manual(image: np.ndarray, title: str = "Draw ROI") -> tuple:
    """
    Draw a bounding box by click-drag. Returns (x, y, w, h) in image pixels.
    Returns (0, 0, 0, 0) if cancelled (C or Esc key).

    Uses raw namedWindow + setMouseCallback to avoid cv2.selectROI's Qt
    window-handler issue when OpenCV is built against a mismatched Qt version.
    """
    state = {"start": None, "end": None, "drawing": False, "done": False, "cancel": False}
    base = image.copy()
    canvas = [base.copy()]

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
            img = base.copy()
            cv2.rectangle(img, state["start"], state["end"], (0, 255, 0), 2)
            canvas[0] = img
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"] = (x, y)
            state["drawing"] = False
            img = base.copy()
            cv2.rectangle(img, state["start"], state["end"], (0, 255, 0), 2)
            canvas[0] = img

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, canvas[0])

    while True:
        cv2.imshow(title, canvas[0])
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord(" ")):   # Enter or Space — confirm
            state["done"] = True
            break
        if key in (ord("c"), ord("C"), 27):  # C or Esc — cancel
            state["cancel"] = True
            break

    cv2.destroyWindow(title)

    if state["cancel"] or state["start"] is None or state["end"] is None:
        return (0, 0, 0, 0)

    x1, y1 = state["start"]
    x2, y2 = state["end"]
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# SAMURAI offline mask segmentation
# ---------------------------------------------------------------------------

class SamuraiPostProcessor(PostProcessor):
    """
    Runs SAMURAI (SAM2-based) offline on all episode frames to produce
    per-frame binary object masks.

    Unlike the live ROS node, all frames are given to SAM2 at once so the
    model can use full temporal context (offline / batch mode), which is
    both more accurate and avoids real-time latency constraints.

    A bounding box prompt on frame 0 is required; it is collected on the
    main thread by the collector before this processor is called.

    Adds to the episode:
        mask : list[np.ndarray (D,D) uint8]  — 0/255 binary mask per frame
    """

    def __init__(
        self,
        checkpoint: str = SAMURAI_CHECKPOINT_DEFAULT,
        config: str = SAMURAI_CONFIG_DEFAULT,
        samurai_path: str = SAMURAI_PATH_DEFAULT,
    ):
        samurai_path = os.path.expanduser(samurai_path)
        checkpoint = os.path.expanduser(checkpoint)
        if samurai_path not in sys.path:
            sys.path.insert(0, samurai_path)

        from sam2.build_sam import build_sam2_video_predictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        self._bbox: Optional[np.ndarray] = None

    def prepare(self, episode: dict) -> None:
        """Show frame 0 for the user to draw a bounding box (main-thread, blocking)."""
        frame0 = episode["img"][0]
        D = frame0.shape[0]
        display_dim = max(D, 480)

        vis = cv2.resize(frame0, (display_dim, display_dim), interpolation=cv2.INTER_NEAREST)
        r = _select_roi_manual(vis, "SAMURAI: drag bbox, Enter to confirm, C to cancel")

        cv2.destroyAllWindows()
        if r[2] > 0 and r[3] > 0:
            scale = D / display_dim
            self._bbox = np.array(
                [r[0] * scale, r[1] * scale,
                 (r[0] + r[2]) * scale, (r[1] + r[3]) * scale],
                dtype=np.float32,
            )
        else:
            self._bbox = None

    def process(self, episode: dict) -> dict:
        """Run SAMURAI on all episode frames using the bbox set by prepare()."""
        if self._bbox is None:
            raise RuntimeError("SamuraiPostProcessor: prepare() was cancelled or not called")

        # FoundationPose calls torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # as a global side effect.  SAM2's load_video_frames_from_numpy creates
        # img_mean_t/img_std_t via torch.tensor() — which then land on CUDA — while
        # torch.from_numpy() always returns a CPU tensor, causing a device mismatch
        # on every episode after the first.  Reset here before init_state is called.
        torch.set_default_tensor_type(torch.FloatTensor)

        frames = episode["img"]   # list of (D,D,3) uint8 BGR
        N = len(frames)
        H, W = frames[0].shape[:2]

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        # Batch init: give SAM2 all frames so it can use full temporal context.
        # Keep video frames on the model device (no offload_video_to_cpu) so that
        # SAM2's normalisation constants and the frame tensors are always co-located.
        # offload_state_to_cpu keeps the per-frame feature cache on CPU to save VRAM.
        inference_state = self._predictor.init_state(
            frames=rgb_frames,
            offload_state_to_cpu=True,
        )

        self._predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=0,
            box=self._bbox,
        )

        masks = [np.zeros((H, W), dtype=np.uint8)] * N
        for frame_idx, _, video_res_masks in tqdm(
            self._predictor.propagate_in_video(inference_state),
            total=N, desc="SAMURAI", unit="frame",
        ):
            masks[frame_idx] = (
                (video_res_masks[0, 0].cpu().numpy() > 0.0).astype(np.uint8) * 255
            )

        self._predictor.reset_state(inference_state)
        return {"mask": masks}


# ---------------------------------------------------------------------------
# SAMURAI + FoundationPose implementation
# ---------------------------------------------------------------------------

class SamuraiFoundationPoseProcessor(PostProcessor):
    """
    Estimates 6D object pose via FoundationPose.

    If the episode already contains per-frame masks (e.g. from SamuraiPostProcessor),
    the frame with the largest mask blob is used as the registration anchor.  If no
    masks are present, a SAM2 image predictor is run on frame 0 with a fixed query
    point to obtain a single registration mask, avoiding the need to run the full
    SAMURAI video segmentation first.

    Adds to the episode:
        obj_pose_4x4 : list[np.ndarray (4,4)]  — object-in-camera SE(3)
    """

    def __init__(
        self,
        mesh_path: str,
        apply_scale: float = 1.0,
        fp_ros_path: str = FP_ROS_PATH_DEFAULT,
        checkpoint: str = SAMURAI_CHECKPOINT_DEFAULT,
        config: str = SAMURAI_CONFIG_DEFAULT,
        samurai_path: str = SAMURAI_PATH_DEFAULT,
        query_point: Optional[tuple] = None,
    ):
        # --- FoundationPose ---
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

        # --- SAM2 image predictor (for mask-free initialization) ---
        samurai_path = os.path.expanduser(samurai_path)
        checkpoint = os.path.expanduser(checkpoint)
        if samurai_path not in sys.path:
            sys.path.insert(0, samurai_path)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam2_predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
        self._query_point: tuple = query_point
        self._last_query_point: Optional[tuple] = None  # set from previous episode's pose

    def process(self, episode: dict) -> dict:
        frames = episode["img"]    # list of (D,D,3) uint8 BGR
        depths = episode["depth"]  # list of (D,D) float32 metres
        masks = episode.get("mask")  # list of (D,D) uint8, or None
        K = episode.get("camera_K")
        if K is None:
            raise ValueError("Episode has no 'camera_K' — was it recorded with depth?")
        N = len(frames)
        if N == 0:
            return {"obj_pose_4x4": []}

        # Determine registration anchor and its mask.
        # If per-frame SAMURAI masks are available, pick the frame with the largest
        # blob as the anchor.  Otherwise fall back to SAM2 on frame 0.
        has_masks = masks is not None and max(int((m > 0).sum()) for m in masks) > 0
        if has_masks:
            blob_sizes = [int((m > 0).sum()) for m in masks]
            anchor = _find_registration_anchors(blob_sizes)[0]
            reg_mask = masks[anchor] > 0
        else:
            anchor = 0
            reg_mask = self._sam2_init_mask(frames[0])

        poses: list = [None] * N

        rgb = cv2.cvtColor(frames[anchor], cv2.COLOR_BGR2RGB)
        poses[anchor] = self._est.register(
            K=K, rgb=rgb, depth=depths[anchor],
            ob_mask=reg_mask, iteration=10,
        )
        pose_at_anchor = self._est.pose_last.clone()

        # Forward: anchor+1 to end of episode
        for i in trange(
            anchor + 1, N,
            desc=f"  fwd [{anchor}→{N}]", leave=False, unit="frame",
        ):
            rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=10)

        self._est.pose_last = pose_at_anchor.clone()
        for i in trange(
            anchor - 1, -1, -1,
            desc=f"  bwd [{anchor}→0]", leave=False, unit="frame",
        ):
            rgb_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            poses[i] = self._est.track_one(rgb=rgb_i, depth=depths[i], K=K, iteration=10)

        _visualize_anchors(frames, masks or [], poses, [anchor], K)

        # Cache the projected 2D centre of the last valid pose for the next episode.
        last_pose = next((p for p in reversed(poses) if p is not None), None)
        if last_pose is not None:
            self._last_query_point = _project_pose_to_frac(last_pose, K, frames[0].shape)

        return {"obj_pose_4x4": poses}

    def _sam2_init_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a bool mask for frame using SAM2 with a query point.

        Priority:
          1. Explicit query_point set at construction — always used when provided.
          2. Projected centre of the last valid pose from the previous episode.
          3. Image centre (0.5, 0.5).
        """
        if self._query_point is not None:
            qp = self._query_point
        elif self._last_query_point is not None:
            qp = self._last_query_point
        else:
            qp = (0.5, 0.5)
        H, W = frame.shape[:2]
        x = int(qp[0] * W)
        y = int(qp[1] * H)
        torch.set_default_tensor_type(torch.FloatTensor)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            self._sam2_predictor.set_image(rgb)
            masks_out, scores, _ = self._sam2_predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        return masks_out[int(np.argmax(scores))].astype(bool)


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




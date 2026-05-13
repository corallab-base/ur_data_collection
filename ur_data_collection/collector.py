#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import select
import termios
import tty
import threading
import traceback
import time
import numpy as np
from typing import Optional
from collections import defaultdict

import pickle
import cv2
from datetime import datetime

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController
from control_msgs.action import FollowJointTrajectory

from cv_bridge import CvBridge

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_pose_stamped

from goc_demo import robotiq

from ur_data_collection.post_processor import (
    SamuraiFoundationPoseProcessor,
    SamuraiPostProcessor,
    deduplicate_episode,
)


WORLD_FRAME = "world"

KEYBINDINGS = (
    "\n"
    "  r  — start / stop recording\n"
    "  g  — open / close gripper\n"
    "  h  — return to home position (captured at startup)\n"
    "  p  — toggle live camera preview\n"
    "  v  — play back most recent episode  (Q / ESC inside window to stop)\n"
    "  a  — annotate most recent episode with post-processor\n"
    "  s  — save most recent episode to disk\n"
    "  q  — quit\n"
)

DISPLAY_DIM_DEFAULT = 480
LIVE_HZ = 30


class CollectorNode(Node):
    """Collects robot demonstration data from several topics."""

    def __init__(self, post_processors=None, save_dir: str = "saved_data"):
        super().__init__("collector_node")

        self._post_processors: list = post_processors or []
        self._annotating = False
        self._awaiting_prepare = False
        self._prepare_episode: Optional[dict] = None
        self._save_dir = save_dir
        self._save_count = 0

        # --- Parameters ---
        self.declare_parameter("pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("mask_topic", "/obj/mask")
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("target_img_dim", 256)
        self.declare_parameter("display_dim", DISPLAY_DIM_DEFAULT)
        self.declare_parameter("home_controller", "scaled_joint_trajectory_controller")
        self.declare_parameter("freedrive_controller", "freedrive_mode_controller")
        self.declare_parameter("home_duration_sec", 5.0)
        # Comma-separated joint angles in radians, e.g. "0.0,-1.57,0.0,-1.57,0.0,0.0".
        # If empty (default), the position is captured from the first /joint_states message.
        self.declare_parameter("home_q", "")

        self.bridge = CvBridge()

        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._image_topic: str = self.get_parameter("image_topic").value
        self._depth_topic: str = self.get_parameter("depth_topic").value
        self._camera_info_topic: str = self.get_parameter("camera_info_topic").value
        self._mask_topic: str = self.get_parameter("mask_topic").value
        self._camera_optical_frame: str = self.get_parameter("camera_optical_frame").value
        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._target_img_dim: int = (
            self.get_parameter("target_img_dim").get_parameter_value().integer_value
        )
        self._display_dim: int = (
            self.get_parameter("display_dim").get_parameter_value().integer_value
        )
        self._home_controller: str = self.get_parameter("home_controller").value
        self._freedrive_controller: str = self.get_parameter("freedrive_controller").value
        self._home_duration_sec: float = float(self.get_parameter("home_duration_sec").value)
        _home_q_str: str = self.get_parameter("home_q").value
        if _home_q_str.strip():
            self._home_q: Optional[np.ndarray] = np.array(
                [float(v) for v in _home_q_str.split(",")]
            )
            self.get_logger().info(f"Home position set from parameter: {self._home_q.tolist()}")
        else:
            self._home_q: Optional[np.ndarray] = None  # captured on first /joint_states

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 20.0")
            self._rate_hz = 20.0

        self._period_sec = 1.0 / self._rate_hz

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- Home position ---
        self._homing = False
        self._switch_ctrl_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self._jtc_client = ActionClient(
            self, FollowJointTrajectory,
            f"/{self._home_controller}/follow_joint_trajectory",
        )

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Sensor state ---
        self._latest_pose: Optional[Pose] = None
        self._latest_q: Optional[np.ndarray] = None
        self._latest_qd: Optional[np.ndarray] = None
        self._latest_eff: Optional[np.ndarray] = None
        self._latest_joint_names: Optional[list] = None
        self._latest_image: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_mask: Optional[np.ndarray] = None
        self._camera_K_adjusted: Optional[np.ndarray] = None
        self._display_image: Optional[np.ndarray] = None

        # --- Display state (owned by main thread) ---
        self._show_live: bool = False
        self._in_playback: bool = False
        self._playback_frames: list = []
        self._playback_masks: list = []
        self._playback_poses: list = []
        self._playback_K: Optional[np.ndarray] = None
        self._playback_idx: int = 0
        self._playback_last_t: float = 0.0

        # --- Subscriptions ---
        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joints, sensor_qos)
        self.create_subscription(Image, self._image_topic, self._on_image, 10)
        self.create_subscription(Image, self._depth_topic, self._on_depth, 10)
        self.create_subscription(CameraInfo, self._camera_info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self._mask_topic, self._on_mask, 10)

        # --- Gripper ---
        self._gripper_open = True
        self._gripper_available = False
        try:
            self._real_gripper = robotiq.RobotiqGripper(disabled=False)
            self._real_gripper.connect("10.168.4.249", 63352)
            self._real_gripper.activate(auto_calibrate=True)
            self._real_gripper.open(speed=2, force=2)
            self._gripper_available = True
        except Exception as e:
            self.get_logger().warn(f"Gripper unavailable: {e}")

        # --- Episode state ---
        self._recording = False
        self._prev_ee_pos: Optional[np.ndarray] = None
        self._current_episode: dict = defaultdict(list)
        self._episodes: list[dict] = []

        # --- Timer ---
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        # --- Keyboard thread ---
        self._running = True
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        self.get_logger().info(f"Collector ready at {self._rate_hz:.1f} Hz")
        print(KEYBINDINGS, flush=True)

    # --- Sensor callbacks ---

    def _on_joints(self, msg: JointState):
        if self._latest_joint_names is None and msg.name:
            self._latest_joint_names = list(msg.name)
        self._latest_q = np.array(msg.position)
        self._latest_qd = np.array(msg.velocity)
        self._latest_eff = np.array(msg.effort)
        if self._home_q is None and len(msg.position) > 0:
            self._home_q = np.array(msg.position)
            self.get_logger().info(f"Home position captured from startup state: {np.round(self._home_q, 3).tolist()}")

    def _on_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_pose = ps_w.pose

    def _on_image(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = cv_img.shape[:2]
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        square = cv_img[start_y:start_y + size, start_x:start_x + size]
        self._latest_image = cv2.resize(
            square,
            (self._target_img_dim, self._target_img_dim),
            interpolation=cv2.INTER_AREA,
        )
        self._display_image = cv2.resize(
            square,
            (self._display_dim, self._display_dim),
            interpolation=cv2.INTER_AREA,
        )

    def _on_depth(self, msg: Image):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")  # uint16
        dm = raw.astype(np.float32) / 1000.0
        h, w = dm.shape[:2]
        size = min(h, w)
        sx, sy = (w - size) // 2, (h - size) // 2
        square = dm[sy:sy + size, sx:sx + size]
        self._latest_depth = cv2.resize(
            square,
            (self._target_img_dim, self._target_img_dim),
            interpolation=cv2.INTER_NEAREST,
        )

    def _on_mask(self, msg: Image):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        h, w = mask.shape[:2]
        size = min(h, w)
        sx, sy = (w - size) // 2, (h - size) // 2
        square = mask[sy:sy + size, sx:sx + size]
        self._latest_mask = cv2.resize(
            square,
            (self._target_img_dim, self._target_img_dim),
            interpolation=cv2.INTER_NEAREST,
        )

    def _on_camera_info(self, msg: CameraInfo):
        if self._camera_K_adjusted is not None:
            return
        K = np.array(msg.k).reshape(3, 3)
        h, w = msg.height, msg.width
        size = min(h, w)
        sx, sy = (w - size) // 2, (h - size) // 2
        s = self._target_img_dim / size
        self._camera_K_adjusted = np.array([
            [K[0, 0] * s, 0.0,          (K[0, 2] - sx) * s],
            [0.0,         K[1, 1] * s,  (K[1, 2] - sy) * s],
            [0.0,         0.0,           1.0],
        ])

    # --- Timer callback ---

    def _on_timer(self):
        if not self._recording:
            return
        if self._latest_pose is None:
            self.get_logger().warn(
                "Recording: waiting for pose", throttle_duration_sec=2.0
            )
            return
        if self._latest_q is None:
            self.get_logger().warn(
                "Recording: waiting for joint states", throttle_duration_sec=2.0
            )
            return
        if self._latest_image is None:
            self.get_logger().warn(
                "Recording: waiting for camera image", throttle_duration_sec=2.0
            )
            return
        if self._latest_depth is None:
            self.get_logger().warn(
                "Recording: waiting for depth image", throttle_duration_sec=2.0
            )
            return

        pose = self._latest_pose

        ee_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        action = ee_pos - self._prev_ee_pos if self._prev_ee_pos is not None else np.zeros(3)
        self._prev_ee_pos = ee_pos

        self._current_episode["img"].append(self._latest_image.copy())
        self._current_episode["depth"].append(self._latest_depth.copy())
        self._current_episode["mask"].append(
            self._latest_mask.copy() if self._latest_mask is not None
            else np.zeros((self._target_img_dim, self._target_img_dim), dtype=np.uint8)
        )
        self._current_episode["q"].append(self._latest_q.copy())
        self._current_episode["qd"].append(self._latest_qd.copy())
        self._current_episode["eff"].append(self._latest_eff.copy())
        self._current_episode["ee_pos"].append(ee_pos)
        self._current_episode["ee_quat_wxyz"].append(np.array([
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]))
        self._current_episode["gripper_pos"].append(
            self._real_gripper.get_current_position() if self._gripper_available else 0
        )
        self._current_episode["action"].append(action)

    # --- Control methods ---

    def _toggle_recording(self):
        if not self._recording:
            self._current_episode = defaultdict(list)
            self._current_episode["camera_K"] = self._camera_K_adjusted  # single 3×3, not a list
            self._current_episode["T_world_camera"] = self._lookup_camera_tf()
            self._current_episode["joint_names"] = self._latest_joint_names
            self._prev_ee_pos = None
            self._recording = True
            self.get_logger().info("Recording STARTED")
        else:
            self._recording = False
            n = len(self._current_episode.get("img", []))
            self._episodes.append(dict(self._current_episode))
            self.get_logger().info(
                f"Recording STOPPED — {n} frames (episode #{len(self._episodes)})"
            )

    def _toggle_gripper(self):
        if not self._gripper_available:
            self.get_logger().warn("Gripper not connected")
            return
        self._real_gripper.toggle()
        self._gripper_open = not self._gripper_open

    def _toggle_live_preview(self):
        self._show_live = not self._show_live
        if not self._show_live:
            cv2.destroyWindow("Live")

    def _start_playback(self):
        if not self._episodes:
            self.get_logger().warn("No completed episodes to preview")
            return
        frames = self._episodes[-1].get("img", [])
        if not frames:
            self.get_logger().warn("Episode has no images")
            return
        ep = self._episodes[-1]
        self._playback_frames = frames
        self._playback_masks = ep.get("mask", [])
        self._playback_poses = ep.get("obj_pose_4x4", [])
        self._playback_K = ep.get("camera_K")
        self._playback_idx = 0
        self._playback_last_t = time.monotonic()
        self._in_playback = True
        self.get_logger().info(
            f"Playback started — {len(frames)} frames  (Q / ESC to stop)"
        )

    def _step_display(self):
        """Must be called from the main thread each iteration."""
        if self._awaiting_prepare and self._prepare_episode is not None:
            self._awaiting_prepare = False
            ep = self._prepare_episode
            self._prepare_episode = None
            for proc in self._post_processors:
                proc.prepare(ep)  # may block (e.g. bbox drawing for SAMURAI)
            threading.Thread(target=self._run_annotation, args=(ep,), daemon=True).start()
            return

        showed = False

        if self._in_playback and self._playback_frames:
            now = time.monotonic()
            if now - self._playback_last_t >= 1.0 / self._rate_hz:
                self._playback_idx = (self._playback_idx + 1) % len(self._playback_frames)
                self._playback_last_t = now
            frame = self._playback_frames[self._playback_idx]
            display = cv2.resize(frame, (self._display_dim, self._display_dim))
            if self._playback_masks and self._playback_idx < len(self._playback_masks):
                m = self._playback_masks[self._playback_idx]
                m_up = cv2.resize(
                    m, (self._display_dim, self._display_dim),
                    interpolation=cv2.INTER_NEAREST,
                )
                overlay = display.copy()
                overlay[m_up > 0] = (0, 200, 0)
                cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
            if (
                self._playback_poses
                and self._playback_K is not None
                and self._playback_idx < len(self._playback_poses)
            ):
                _draw_pose_axes(
                    display,
                    self._playback_poses[self._playback_idx],
                    self._playback_K,
                    self._target_img_dim,
                    self._display_dim,
                )
            cv2.putText(
                display,
                f"{self._playback_idx + 1} / {len(self._playback_frames)}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.imshow("Playback", display)
            showed = True

        if self._show_live:
            img = self._display_image
            if img is not None:
                display = img.copy()
                label = "REC" if self._recording else "IDLE"
                color = (0, 0, 220) if self._recording else (180, 180, 180)
                cv2.putText(
                    display, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )
                cv2.imshow("Live", display)
                showed = True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or ESC closes active windows
            if self._in_playback:
                self._in_playback = False
                cv2.destroyWindow("Playback")
            if self._show_live:
                self._show_live = False
                cv2.destroyWindow("Live")

    def _annotate_episode(self):
        if not self._post_processors:
            self.get_logger().warn("No post-processors configured")
            return
        if not self._episodes:
            self.get_logger().warn("No completed episodes to annotate")
            return
        if self._annotating:
            self.get_logger().warn("Annotation already in progress")
            return
        self._annotating = True
        self._prepare_episode = self._episodes[-1]
        self._awaiting_prepare = True

    def _run_annotation(self, episode: dict):
        try:
            deduplicate_episode(episode)
            for proc in self._post_processors:
                self.get_logger().info(f"{proc.__class__.__name__} started...")
                additions = proc.process(episode)
                episode.update(additions)
                self.get_logger().info(
                    f"{proc.__class__.__name__} complete — keys: {list(additions.keys())}"
                )
        except Exception as e:
            self.get_logger().error(f"Annotation failed: {e}")
            traceback.print_exc()
        finally:
            self._annotating = False

    def _save_episode(self):
        if not self._episodes:
            self.get_logger().warn("No completed episodes to save")
            return
        path = os.path.join(self._save_dir, f"episode_{self._save_count}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._episodes[-1], f)
        self._save_count += 1
        self.get_logger().info(f"Saved episode to {path}")

    # --- Keyboard thread ---

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, ch: str):
        if ch == "r":
            self._toggle_recording()
        elif ch == "g":
            self._toggle_gripper()
        elif ch == "h":
            self._return_to_home()
        elif ch == "p":
            self._toggle_live_preview()
        elif ch == "v":
            self._start_playback()
        elif ch == "a":
            self._annotate_episode()
        elif ch == "s":
            self._save_episode()
        elif ch in ("q", "\x03"):
            self._running = False

    # --- Homing ---

    def _return_to_home(self):
        if self._home_q is None:
            self.get_logger().warn("Home position not yet captured — wait for /joint_states")
            return
        if self._latest_joint_names is None:
            self.get_logger().warn("Joint names not yet received — wait for /joint_states")
            return
        if self._homing:
            self.get_logger().warn("Already homing")
            return
        if self._recording:
            self._toggle_recording()
        self._homing = True
        threading.Thread(target=self._run_home_sequence, daemon=True).start()

    def _run_home_sequence(self):
        try:
            self.get_logger().info(
                f"Homing: deactivating {self._freedrive_controller}, "
                f"activating {self._home_controller}..."
            )
            self._switch_controllers(
                activate=[self._home_controller],
                deactivate=[self._freedrive_controller],
            )
            self.get_logger().info(
                f"Homing: moving to home over {self._home_duration_sec:.1f}s..."
            )
            self._send_home_trajectory()
            self.get_logger().info(
                f"Homing: reactivating {self._freedrive_controller}..."
            )
            self._switch_controllers(
                activate=[self._freedrive_controller],
                deactivate=[self._home_controller],
            )
            self.get_logger().info("Homing complete.")
        except Exception as e:
            self.get_logger().error(f"Home sequence failed: {e}")
            traceback.print_exc()
        finally:
            self._homing = False

    def _switch_controllers(self, activate: list, deactivate: list, timeout_sec: float = 5.0):
        if not self._switch_ctrl_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("/controller_manager/switch_controller service not available")
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        future = self._switch_ctrl_client.call_async(req)
        result = self._wait_future(future, timeout_sec=timeout_sec)
        if not result.ok:
            raise RuntimeError(
                f"switch_controller rejected: activate={activate} deactivate={deactivate}"
            )

    def _send_home_trajectory(self):
        if not self._jtc_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"Action server not available: /{self._home_controller}/follow_joint_trajectory"
            )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self._latest_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = self._home_q.tolist()
        pt.time_from_start = Duration(seconds=self._home_duration_sec).to_msg()
        goal.trajectory.points = [pt]

        goal_future = self._jtc_client.send_goal_async(goal)
        goal_handle = self._wait_future(goal_future, timeout_sec=5.0)
        if not goal_handle.accepted:
            raise RuntimeError("FollowJointTrajectory goal was rejected")

        result_response = self._wait_future(
            goal_handle.get_result_async(),
            timeout_sec=self._home_duration_sec + 10.0,
        )
        if result_response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory execution failed: error_code={result_response.result.error_code}"
            )

    def _wait_future(self, future, timeout_sec: float = 10.0):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"ROS future timed out after {timeout_sec}s")
            time.sleep(0.05)
        return future.result()

    def _lookup_camera_tf(self) -> Optional[np.ndarray]:
        """Return T_world_camera as (4,4) float64, or None if TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, self._camera_optical_frame,
                Time(), timeout=Duration(seconds=1.0),
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = _quat_to_rot(r.x, r.y, r.z, r.w)
            mat[:3, 3] = [t.x, t.y, t.z]
            return mat
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"Could not look up camera→world TF ({self._camera_optical_frame} → "
                f"{WORLD_FRAME}): {e}"
            )
            return None

    # --- TF helpers ---

    def _to_world(
        self, pose_msg: PoseStamped, timeout_sec: float = 0.1
    ) -> Optional[PoseStamped]:
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return pose_msg
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                src_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec),
            )
            return do_transform_pose_stamped(pose_msg, tf)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF ({WORLD_FRAME} <- {src_frame}): {e}")
            return None



def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _draw_pose_axes(
    img: np.ndarray,
    pose: np.ndarray,
    K_recorded: np.ndarray,
    recorded_dim: int,
    display_dim: int,
    axis_len: float = 0.05,
) -> None:
    """Overlay XYZ axes of a 4x4 camera-frame pose onto img (in place).

    K_recorded was built for recorded_dim; img is display_dim × display_dim,
    so K is rescaled before projecting.
    """
    s = display_dim / recorded_dim
    K = K_recorded * s
    K[2, 2] = 1.0

    origin = pose[:3, 3]
    tips = origin + axis_len * pose[:3, :3].T  # rows: X-tip, Y-tip, Z-tip

    def proj(pt):
        p = K @ pt
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    o = proj(origin)
    colors = [(0, 0, 220), (0, 220, 0), (220, 0, 0)]  # X=red, Y=green, Z=blue (BGR)
    for tip, color in zip(tips, colors):
        cv2.line(img, o, proj(tip), color, 2, cv2.LINE_AA)
    cv2.circle(img, o, 3, (255, 255, 255), -1)


def main(args=None):
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("saved_data", session_ts)
    os.makedirs(save_dir, exist_ok=True)
    rclpy.init(args=args)

    post_processors = []

    # checkpoint = os.environ.get("SAMURAI_CHECKPOINT", "")
    # if checkpoint:

    # post_processors.append(SamuraiPostProcessor())

    # mesh_path = os.environ.get("MESH_PATH", "")
    # if mesh_path:
    #     post_processors.append(SamuraiFoundationPoseProcessor(mesh_path=mesh_path))

    node = CollectorNode(post_processors=post_processors, save_dir=save_dir)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    frame_time = 1.0 / LIVE_HZ
    try:
        while rclpy.ok() and node._running:
            t0 = time.monotonic()
            node._step_display()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, frame_time - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

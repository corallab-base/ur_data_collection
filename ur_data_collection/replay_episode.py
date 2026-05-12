#!/usr/bin/env python3
"""ROS 2 node that replays a recorded episode for visualisation in RViz.

Publishes:
  /joint_states                — sensor_msgs/JointState  (robot pose per frame)
  /tf                          — dynamic: world → <object_frame_id>
  /tf_static                   — static:  world → camera_color_optical_frame
  /replay/pose_marker          — visualization_msgs/MarkerArray (axes triad)

Run via the companion launch file:
  ros2 launch ur_data_collection replay_episode.launch.py \\
      episode_path:=/abs/path/episode_0.pkl rate_hz:=10.0
"""

from __future__ import annotations

import pickle
import sys
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

# Fallback joint names for UR5e when the episode pre-dates name storage.
# These are the names published by joint_state_broadcaster in alphabetical order.
_UR5E_JOINT_NAMES = [
    'elbow_joint',
    'shoulder_lift_joint',
    'shoulder_pan_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def _mat_to_transform_stamped(
    mat: np.ndarray,
    parent: str,
    child: str,
    stamp,
) -> TransformStamped:
    """Convert a 4×4 SE(3) matrix to a TransformStamped."""
    from geometry_msgs.msg import Quaternion, Vector3 as V3
    from scipy.spatial.transform import Rotation

    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(mat[0, 3])
    ts.transform.translation.y = float(mat[1, 3])
    ts.transform.translation.z = float(mat[2, 3])
    q = Rotation.from_matrix(mat[:3, :3]).as_quat()  # (x, y, z, w)
    ts.transform.rotation.x = float(q[0])
    ts.transform.rotation.y = float(q[1])
    ts.transform.rotation.z = float(q[2])
    ts.transform.rotation.w = float(q[3])
    return ts


def _axes_marker_array(
    pose: np.ndarray,
    frame_id: str,
    stamp,
    ns: str = "object_axes",
    axis_len: float = 0.05,
    axis_diam: float = 0.005,
) -> MarkerArray:
    """Return a MarkerArray with three arrow markers (X=red, Y=green, Z=blue)."""
    origin = pose[:3, 3].astype(float)
    axes = pose[:3, :3].T  # rows: X, Y, Z directions
    colors = [
        ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0),
        ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0),
        ColorRGBA(r=0.1, g=0.1, b=0.9, a=1.0),
    ]
    markers = []
    for i, (direction, color) in enumerate(zip(axes, colors)):
        tip = origin + axis_len * direction
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = i
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale = Vector3(x=axis_diam, y=axis_diam * 2, z=0.0)
        m.color = color
        from geometry_msgs.msg import Point
        p_start = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        p_end = Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]))
        m.points = [p_start, p_end]
        markers.append(m)
    return MarkerArray(markers=markers)


class ReplayEpisodeNode(Node):

    def __init__(self):
        super().__init__('replay_episode')

        self.declare_parameter('episode_path', '')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('object_frame_id', 'object')
        self.declare_parameter('tf_prefix', 'replay_')

        episode_path = self.get_parameter('episode_path').value
        rate_hz = float(self.get_parameter('rate_hz').value)
        self._loop = bool(self.get_parameter('loop').value)
        self._object_frame_id = self.get_parameter('object_frame_id').value
        tf_prefix = self.get_parameter('tf_prefix').value

        if not episode_path:
            self.get_logger().fatal("'episode_path' parameter is required but not set.")
            raise RuntimeError("episode_path not set")

        self.get_logger().info(f"Loading episode from {episode_path} …")
        with open(episode_path, 'rb') as f:
            episode = pickle.load(f)
        self.get_logger().info("Episode loaded.")

        # --- Joint state data ---
        self._q_list: list = episode.get('q', [])
        raw_names = episode.get('joint_names') or _UR5E_JOINT_NAMES
        # Apply tf_prefix so names match the prefixed URDF joints in robot_state_publisher.
        self._joint_names: list[str] = [tf_prefix + n for n in raw_names]
        if not self._q_list:
            self.get_logger().warn("Episode has no 'q' data — joint states will not be published.")

        # --- Object poses (world frame) ---
        self._world_poses: list[Optional[np.ndarray]] = self._resolve_world_poses(episode)

        # --- Camera static TF ---
        T_world_cam: Optional[np.ndarray] = episode.get('T_world_camera')

        # --- Publishers / broadcasters ---
        self._js_pub = self.create_publisher(JointState, '/fake/joint_states', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/replay/pose_marker', 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_broadcaster = StaticTransformBroadcaster(self)

        if T_world_cam is not None:
            stamp = self.get_clock().now().to_msg()
            static_tf = _mat_to_transform_stamped(
                T_world_cam, 'world', 'camera_color_optical_frame', stamp)
            self._static_broadcaster.sendTransform([static_tf])
            self.get_logger().info("Published static TF: world → camera_color_optical_frame")
        else:
            self.get_logger().warn("Episode has no 'T_world_camera' — camera TF not published.")

        n = len(self._q_list)
        self.get_logger().info(
            f"Replaying {n} frames at {rate_hz:.1f} Hz "
            f"({'looping' if self._loop else 'once'})."
        )

        self._frame_idx = 0
        self._n_frames = n
        self._done = False
        self.create_timer(1.0 / rate_hz, self._step)

    # ------------------------------------------------------------------ #

    def _resolve_world_poses(self, episode: dict) -> list[Optional[np.ndarray]]:
        """Return per-frame 4×4 world-frame poses, or empty list if unavailable."""
        if 'obj_pose_4x4_world' in episode:
            return list(episode['obj_pose_4x4_world'])

        cam_poses = episode.get('obj_pose_4x4', [])
        T_world_cam = episode.get('T_world_camera')
        if cam_poses and T_world_cam is not None:
            self.get_logger().info(
                "No 'obj_pose_4x4_world' found; computing from 'obj_pose_4x4' + T_world_camera.")
            return [(T_world_cam @ p).astype(np.float64) for p in cam_poses]

        self.get_logger().warn(
            "No object pose data found — object TF and markers will not be published.")
        return []

    def _step(self):
        if self._done:
            return

        if self._n_frames == 0:
            return

        stamp = self.get_clock().now().to_msg()
        idx = self._frame_idx

        # --- Joint states ---
        if idx < len(self._q_list):
            js = JointState()
            js.header.stamp = stamp
            js.name = self._joint_names
            js.position = [float(v) for v in self._q_list[idx]]
            self._js_pub.publish(js)

        # --- Object TF + marker ---
        if self._world_poses and idx < len(self._world_poses):
            pose = self._world_poses[idx]
            if pose is not None:
                tf = _mat_to_transform_stamped(pose, 'world', self._object_frame_id, stamp)
                self._tf_broadcaster.sendTransform(tf)
                markers = _axes_marker_array(pose, 'world', stamp)
                self._marker_pub.publish(markers)

        # --- Advance ---
        self._frame_idx += 1
        if self._frame_idx >= self._n_frames:
            if self._loop:
                self._frame_idx = 0
            else:
                self._done = True
                self.get_logger().info("Replay finished.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ReplayEpisodeNode()
        rclpy.spin(node)
    except (RuntimeError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some useful math functions."""

from typing import Tuple, Optional, Union

import jax
from jax import custom_jvp
from jax import numpy as jp
import numpy as np

def quat_from_mat2(mat_2: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Convert rotation matrices (given by first two columns) to quaternions.

    Args:
        mat_2: Rotation matrix first two columns, shape (..., 3, 2).
               It should come from a proper rotation matrix (orthonormal columns).

    Returns:
        quaternions in (w, x, y, z), shape (..., 4).
    """
    # mat_2[..., :, 0] and mat_2[..., :, 1] are the first two columns
    c0 = mat_2[..., :, 0]  # (..., 3)
    c1 = mat_2[..., :, 1]  # (..., 3)

    # Reconstruct third column via right-handed basis
    # If your convention is the opposite, swap to jp.cross(c1, c0)
    c2 = jp.cross(c0, c1)  # (..., 3)

    # Stack back to full rotation matrix R (..., 3, 3)
    R = jp.stack([c0, c1, c2], axis=-1)

    # Diagonal elements
    r00 = R[..., 0, 0]
    r11 = R[..., 1, 1]
    r22 = R[..., 2, 2]

    trace = r00 + r11 + r22

    # Branch 1: trace > 0
    s_tr = jp.sqrt(jp.maximum(trace + 1.0, eps)) * 2.0  # 4 * qw
    qw_tr = 0.25 * s_tr
    qx_tr = (R[..., 2, 1] - R[..., 1, 2]) / s_tr
    qy_tr = (R[..., 0, 2] - R[..., 2, 0]) / s_tr
    qz_tr = (R[..., 1, 0] - R[..., 0, 1]) / s_tr

    # Branch 2: r00 is largest
    s_x = jp.sqrt(jp.maximum(1.0 + r00 - r11 - r22, eps)) * 2.0  # 4 * qx
    qw_x = (R[..., 2, 1] - R[..., 1, 2]) / s_x
    qx_x = 0.25 * s_x
    qy_x = (R[..., 0, 1] + R[..., 1, 0]) / s_x
    qz_x = (R[..., 0, 2] + R[..., 2, 0]) / s_x

    # Branch 3: r11 is largest
    s_y = jp.sqrt(jp.maximum(1.0 + r11 - r00 - r22, eps)) * 2.0  # 4 * qy
    qw_y = (R[..., 0, 2] - R[..., 2, 0]) / s_y
    qx_y = (R[..., 0, 1] + R[..., 1, 0]) / s_y
    qy_y = 0.25 * s_y
    qz_y = (R[..., 1, 2] + R[..., 2, 1]) / s_y

    # Branch 4: r22 is largest
    s_z = jp.sqrt(jp.maximum(1.0 + r22 - r00 - r11, eps)) * 2.0  # 4 * qz
    qw_z = (R[..., 1, 0] - R[..., 0, 1]) / s_z
    qx_z = (R[..., 0, 2] + R[..., 2, 0]) / s_z
    qy_z = (R[..., 1, 2] + R[..., 2, 1]) / s_z
    qz_z = 0.25 * s_z

    # Conditions for each branch
    cond_tr = trace > 0.0
    cond_x = (trace <= 0.0) & (r00 > r11) & (r00 > r22)
    cond_y = (trace <= 0.0) & ~(cond_x) & (r11 > r22)
    cond_z = ~(cond_tr | cond_x | cond_y)

    def select(a_tr, a_x, a_y, a_z):
        return jp.where(
            cond_tr, a_tr,
            jp.where(
                cond_x, a_x,
                jp.where(cond_y, a_y, a_z),
            ),
        )

    qw = select(qw_tr, qw_x, qw_y, qw_z)
    qx = select(qx_tr, qx_x, qx_y, qx_z)
    qy = select(qy_tr, qy_x, qy_y, qy_z)
    qz = select(qz_tr, qz_x, qz_y, qz_z)

    quat = jp.stack([qw, qx, qy, qz], axis=-1)

    # normalize once more to avoid numerical error
    quat = quat / (jp.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    return quat


def matrix_from_quat(quaternions: jax.Array) -> jax.Array:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).
    """
    # r, i, j, k = torch.unbind(quaternions, -1)
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]

    # two_s = 2.0 / (quaternions * quaternions).sum(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # stack the 9 matrix entries, exactly as in the PyTorch version
    o = jp.stack(
        (
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))

def subtract_frame_transforms(
    t01: jax.Array,
    q01: jax.Array,
    t02: jax.Array | None = None,
    q02: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: T_12 = T_01^{-1} × T_02,
    where T_AB is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape (N, 3).
        q01: Quaternion of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape (N, 3) or None.
        q02: Quaternion of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape (N, 4) or None.

    Returns:
        (t12, q12): position and orientation of frame 2 w.r.t. frame 1.
        Shapes: (N, 3), (N, 4).
    """
    # compute orientation: q10 = q01^{-1}, q12 = q10 * q02 (or q10 if q02 is None)
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul_batch(q10, q02)
    else:
        q12 = q10

    # compute translation: t12 = q10 ∘ (t02 - t01)  (or q10 ∘ (-t01) if t02 is None)
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)

    return t12, q12

def quat_apply(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: (..., 4) quaternion in (w, x, y, z).
        vec:  (..., 3) vector in (x, y, z).

    Returns:
        (..., 3) rotated vector.
    """
    # svec.shape
    shape = vec.shape

    # quat = quat.reshape(-1, 4); vec = vec.reshape(-1, 3)
    quat_flat = quat.reshape(-1, 4)
    vec_flat = vec.reshape(-1, 3)

    # xyz = quat[:, 1:]
    xyz = quat_flat[..., 1:]  # (N, 3)

    # t = xyz.cross(vec, dim=-1) * 2
    t = jp.cross(xyz, vec_flat, axis=-1) * 2.0  # (N, 3)

    # vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)
    w = quat_flat[..., 0:1]
    rotated = vec_flat + w * t + jp.cross(xyz, t, axis=-1)

    return rotated.reshape(shape)

def quat_normalize(q: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Normalize a quaternion.

    Args:
        q: (..., 4) quaternion in (w, x, y, z).

    Returns:
        (..., 4) normalized quaternion in (w, x, y, z).
    """
    norm = jp.linalg.norm(q, axis=-1, keepdims=True)
    return q / jp.maximum(norm, eps)


def yaw_quat(quat: jax.Array) -> jax.Array:
    """Extract the yaw component of a quaternion.

    Args:
        quat: (..., 4) in (w, x, y, z).

    Returns:
        (..., 4) pure yaw quaternion in (w, x, y, z).
    """
    shape = quat.shape

    quat_yaw = quat.reshape(-1, 4)

    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]

    # yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw = jp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    # quat_yaw[:] = 0.0
    # quat_yaw[:, 3] = torch.sin(yaw / 2)
    # quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = jp.zeros_like(quat_yaw)
    quat_yaw = quat_yaw.at[:,3].set(jp.sin(yaw * 0.5))
    quat_yaw = quat_yaw.at[:,0].set(jp.cos(yaw * 0.5))

    quat_yaw = quat_normalize(quat_yaw)

    return quat_yaw.reshape(shape)

def quat_inv(q: jp.ndarray) -> jp.ndarray:
    """Calculates the inverse of quaternion q.

    Args:
    q: (4,) quaternion [w, x, y, z]

    Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return q * jp.array([1, -1, -1, -1])

def quat_conjugate_jax(q: jax.Array) -> jax.Array:
    """q: (..., 4) in (w, x, y, z)"""
    return jp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def axis_angle_from_quat_jax(q: jax.Array, eps: float = 1e-6) -> jax.Array:
    sign = jp.where(q[0:1] < 0.0, -1.0, 1.0)
    q = q * sign

    v = q[1:]
    mag = jp.linalg.norm(v, axis=-1)               # |v|
    half_angle = jp.arctan2(mag, q[0])        # θ/2
    angle = 2.0 * half_angle                       # θ

    sin_half = jp.sin(half_angle)
    sin_half_over_angle = jp.where(
        jp.abs(angle) > eps,
        sin_half / angle,
        0.5 - angle * angle / 48.0,
    )

    return v / sin_half_over_angle

def axis_angle_from_quat_batch_jax(q: jax.Array, eps: float = 1e-6) -> jax.Array:
    sign = jp.where(q[:, 0:1] < 0.0, -1.0, 1.0)
    q = q * sign

    v = q[:, 1:]
    mag = jp.linalg.norm(v, axis=-1)               # |v|
    half_angle = jp.arctan2(mag, q[:, 0])        # θ/2
    angle = 2.0 * half_angle                       # θ

    sin_half = jp.sin(half_angle)
    sin_half_over_angle = jp.where(
        jp.abs(angle) > eps,
        sin_half / angle,
        0.5 - angle * angle / 48.0,
    )

    return v / sin_half_over_angle[:, None]

def quat_error_magnitude_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    quat_diff = quat_mul_batch(q1, quat_conjugate_jax(q2))
    axis_angle = axis_angle_from_quat_jax(quat_diff)
    return jp.linalg.norm(axis_angle, axis=-1)


def quat_error_magnitude_batch_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    quat_diff = quat_mul_batch(q1, quat_conjugate_jax(q2))
    axis_angle = axis_angle_from_quat_batch_jax(quat_diff)
    return jp.linalg.norm(axis_angle, axis=-1)

def compute_body_motion_relative_w(ref_anchor_pos_w: jax.Array, ref_anchor_quat_w: jax.Array, actual_anchor_pos_w: jax.Array, actual_anchor_quat_w: jax.Array, ref_body_quat_w: jax.Array, ref_body_pos_w: jax.Array, num_track_body: int) -> Tuple[jax.Array, jax.Array]:

    # anchor_pos_w:          (E, 3)   -> (E, 1, 3) -> (E, B, 3)
    ref_anchor_pos_w_repeat = jp.repeat(ref_anchor_pos_w[None, :], num_track_body, axis=0)
    ref_anchor_quat_w_repeat = jp.repeat(ref_anchor_quat_w[None, :], num_track_body, axis=0)
    robot_anchor_pos_w_repeat = jp.repeat(actual_anchor_pos_w[None, :], num_track_body, axis=0)
    robot_anchor_quat_w_repeat = jp.repeat(actual_anchor_quat_w[None, :], num_track_body, axis=0)

    robot_anchor_pos_w_repeat = robot_anchor_pos_w_repeat.at[..., 2].set(ref_anchor_pos_w_repeat[..., 2])

    delta_ori_w = yaw_quat(
        quat_mul_batch(robot_anchor_quat_w_repeat, quat_inv(ref_anchor_quat_w_repeat))
    )

    ref_body_quat_relative_w = quat_mul_batch(delta_ori_w, ref_body_quat_w)

    ref_body_pos_relative_w = robot_anchor_pos_w_repeat + quat_apply(
        delta_ori_w,
        ref_body_pos_w - ref_anchor_pos_w_repeat,
    )
    return ref_body_pos_relative_w, ref_body_quat_relative_w

def rotate(vec: jp.ndarray, quat: jp.ndarray):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

    Returns:
    ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError('vec must have no batch dimensions.')
    s, u = quat[0], quat[1:]
    r = 2 * (jp.dot(u, vec) * u) + (s * s - jp.dot(u, u)) * vec
    r = r + 2 * s * jp.cross(u, vec)
    return r

def inv_rotate(vec: jp.ndarray, quat: jp.ndarray):
    """Rotates a vector vec by an inverted unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by the inverse of quat.
    """
    return rotate(vec, quat_inv(quat))


def rotate_np(vec: np.ndarray, quat: np.ndarray):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError('vec must have no batch dimensions.')
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def ang_to_quat(ang: jp.ndarray):
    """Converts angular velocity to a quaternion.

    Args:
      ang: (3,) angular velocity

    Returns:
      A rotation quaternion.
    """
    return jp.array([0, ang[0], ang[1], ang[2]])


def quat_mul_batch(q1: jax.Array, q2: jax.Array) -> jax.Array:
    # q1, q2: (..., 4)
    w1, x1, y1, z1 = jp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jp.split(q2, 4, axis=-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return jp.concatenate([w, x, y, z], axis=-1)


def quat_mul(u: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return jp.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])


def quat_mul_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return np.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])


def quat_inv(q: jp.ndarray) -> jp.ndarray:
    """Calculates the inverse of quaternion q.

    Args:
      q: (4,) quaternion [w, x, y, z]

    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return q * jp.array([1, -1, -1, -1])


def quat_rot_axis(axis: jp.ndarray, angle: jp.ndarray) -> jp.ndarray:
    """Provides a quaternion that describes rotating around axis v by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around v by angle
    """
    qx = axis[0] * jp.sin(angle / 2)
    qy = axis[1] * jp.sin(angle / 2)
    qz = axis[2] * jp.sin(angle / 2)
    qw = jp.cos(angle / 2)
    return jp.array([qw, qx, qy, qz])


def quat_to_3x3(q: jp.ndarray) -> jp.ndarray:
    """Converts quaternion to 3x3 rotation matrix."""
    d = jp.dot(q, q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return jp.array([
        jp.array([1 - (yy + zz), xy - wz, xz + wy]),
        jp.array([xy + wz, 1 - (xx + zz), yz - wx]),
        jp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
    ])


def quat_from_3x3(m: jp.ndarray) -> jp.ndarray:
    """Converts 3x3 rotation matrix to quaternion."""
    w = jp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    x = (m[2][1] - m[1][2]) / (w * 4)
    y = (m[0][2] - m[2][0]) / (w * 4)
    z = (m[1][0] - m[0][1]) / (w * 4)
    return jp.array([w, x, y, z])


def quat_mul_ang(q: jp.ndarray, ang: jp.ndarray) -> jp.ndarray:
    """Multiplies a quat by an angular velocity."""
    mat = jp.array([
        [-q[2], q[1], -q[0], q[3]],
        [-q[3], q[0], q[1], -q[2]],
        [-q[0], -q[3], q[2], q[1]],
    ])
    return jp.dot(ang, mat)


def signed_angle(
    axis: jp.ndarray, ref_p: jp.ndarray, ref_c: jp.ndarray
) -> jp.ndarray:
    """Calculates the signed angle between two vectors along an axis.

    Args:
      axis: (3,) common axis around which to calculate change in angle
      ref_p: (3,) vector pointing at 0-degrees offset in the parent's frame
      ref_c: (3,) vector pointing at 0-degrees offset in the child's frame

    Returns:
      The signed angle between two parts.
    """
    return jp.arctan2(jp.dot(jp.cross(ref_p, ref_c), axis), jp.dot(ref_p, ref_c))


@custom_jvp
def safe_arccos(x: jp.ndarray) -> jp.ndarray:
    """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
    return jp.arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = -x_dot / jp.sqrt(1.0 - jp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


@custom_jvp
def safe_arcsin(x: jp.ndarray) -> jp.ndarray:
    """Trigonometric inverse sine, element-wise with safety clipping in grad."""
    return jp.arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arcsin(x)
    tangent_out = x_dot / jp.sqrt(1.0 - jp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


def inv_3x3(m) -> jp.ndarray:
    """Inverse specialized for 3x3 matrices."""
    det = jp.linalg.det(m)
    adjugate = jp.array([
        [
            m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
            m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
            m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
        ],
        [
            m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2],
            m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
            m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
        ],
        [
            m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],
            m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
            m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
        ],
    ])
    return adjugate / (det + 1e-10)


def orthogonals(a: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns orthogonal vectors `b` and `c`, given a normal vector `a`."""
    y, z = jp.array([0, 1, 0]), jp.array([0, 0, 1])
    b = jp.where((-0.5 < a[1]) & (a[1] < 0.5), y, z)
    b = b - a * a.dot(b)
    # make b a normal vector. however if a is a zero vector, zero b as well.
    b = normalize(b)[0] * jp.any(a)
    return b, jp.cross(a, b)


def solve_pgs(a: jp.ndarray, b: jp.ndarray, num_iters: int) -> jp.ndarray:
    """Projected Gauss-Seidel solver for a MLCP defined by matrix A and vector b.
    """
    num_rows = b.shape[0]
    x = jp.zeros((num_rows,))

    def get_x(x, xs):
        i, a_i, b_i = xs
        residual = b_i + jp.dot(a_i, x)
        x_i = x[i] - residual / a_i[i]
        x_i = jp.maximum(x_i, 0.0)
        x = x.at[i].set(x_i)

        return x, None

    # TODO: turn this into a scan
    for _ in range(num_iters):
        x, _ = jax.lax.scan(get_x, x, (jp.arange(num_rows), a, b))

    return x


def inv_approximate(
    a: jp.ndarray, a_inv: jp.ndarray, num_iter: int = 10
) -> jp.ndarray:
    """Use Newton-Schulz iteration to solve ``A^-1``.

    Args:
      a: 2D array to invert
      a_inv: approximate solution to A^-1
      num_iter: number of iterations

    Returns:
      A^-1 inverted matrix
    """

    def body_fn(carry, _):
        a_inv, r, err = carry
        a_inv_next = a_inv @ (np.eye(a.shape[0]) + r)
        r_next = np.eye(a.shape[0]) - a @ a_inv_next
        err_next = jp.linalg.norm(r_next)
        a_inv_next = jp.where(err_next < err, a_inv_next, a_inv)
        return (a_inv_next, r_next, err_next), None

    # ensure ||I - X0 @ A|| < 1, in order to guarantee convergence
    r0 = jp.eye(a.shape[0]) - a @ a_inv
    a_inv = jp.where(jp.linalg.norm(r0) > 1, 0.5 * a.T / jp.trace(a @ a.T), a_inv)
    (a_inv, _, _), _ = jax.lax.scan(body_fn, (a_inv, r0, 1.0), None, num_iter)

    return a_inv


def safe_norm(
    x: jp.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> jp.ndarray:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linal.norm(0) see
    https://github.com/google/jax/issues/3058 for details
    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """

    is_zero = jp.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = jp.where(is_zero, jp.ones_like(x), x)
    n = jp.linalg.norm(x, axis=axis)
    n = jp.where(is_zero, 0.0, n)
    return n


def normalize(
    x: jp.ndarray, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Normalizes an array.

    Args:
      x: A jnp.array
      axis: The axis along which to compute the norm

    Returns:
      A tuple of (normalized array x, the norm).
    """
    norm = safe_norm(x, axis=axis)
    n = x / (norm + 1e-6 * (norm == 0.0))
    return n, norm


def from_to(v1: jp.ndarray, v2: jp.ndarray) -> jp.ndarray:
    """Calculates the quaternion that rotates unit vector v1 to unit vector v2."""
    xyz = jp.cross(v1, v2)
    w = 1.0 + jp.dot(v1, v2)
    rnd = jax.random.uniform(jax.random.PRNGKey(0), (3,))
    v1_o = rnd - jp.dot(rnd, v1) * v1
    xyz = jp.where(w < 1e-6, v1_o, xyz)
    rot = jp.append(w, xyz)
    return rot / jp.linalg.norm(rot)


def euler_to_quat(v: jp.ndarray) -> jp.ndarray:
    """Converts euler rotations in degrees to quaternion."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    c1, c2, c3 = jp.cos(v * jp.pi / 360)
    s1, s2, s3 = jp.sin(v * jp.pi / 360)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return jp.array([w, x, y, z])


def quat_to_euler(q: jp.ndarray) -> jp.ndarray:
    """Converts quaternions to euler rotations in radians."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''

    z = jp.arctan2(
        -2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2],
    )
    # TODO: Investigate why quaternions go so big we need to clip.
    y = safe_arcsin(jp.clip(2 * q[1] * q[3] + 2 * q[0] * q[2], -1.0, 1.0))
    x = jp.arctan2(
        -2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0],
    )

    return jp.array([x, y, z])


def eulerzyx_to_quat(v: jp.ndarray) -> jp.ndarray:
    """Converts euler rotations in degrees to quaternion. v is roll-pitch-yaw."""
    # this follows the Tait-Bryan intrinsic rotation formalism: z-y'-x''
    cphi, ctheta, cpsi = jp.cos(v * jp.pi / 360)
    sphi, stheta, spsi = jp.sin(v * jp.pi / 360)
    w = cpsi * ctheta * cphi + spsi * stheta * sphi
    x = cpsi * ctheta * sphi - spsi * stheta * cphi
    y = cpsi * stheta * cphi + spsi * ctheta * sphi
    z = spsi * ctheta * cphi - cpsi * stheta * sphi
    return jp.array([w, x, y, z])


def quat_to_eulerzyx(q: jp.ndarray) -> jp.ndarray:
    """Converts quaternions to euler rotations in radians; returns
    roll-pitch-yaw."""
    # this follows the Tait-Bryan intrinsic rotation formalism: z-y'-x''

    r = jp.arctan2(
        q[2] * q[3] +  q[0] * q[1],
        0.5 - (q[1] * q[1] + q[2] * q[2]),
    )
    p = safe_arcsin(jp.clip(-2*(q[1] * q[3] - q[0] * q[2]), -1.0, 1.0))
    y = jp.arctan2(
        q[1] * q[2] + q[0] * q[3],
        0.5 - (q[2] * q[2] + q[3] * q[3])
    )

    return jp.array([r, p, y])


def vec_quat_mul(u: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Multiplies a vector u and a quaternion v.

    This is a convenience method for multiplying two quaternions when
    one of the quaternions has a 0-value w-part, i.e.:
    quat_mul([0.,a,b,c], [d,e,f,g])

    It is slightly more efficient than constructing a 0-w-part quaternion
    from the vector.

    Args:
      u: (3,) vector representation of the quaternion (0.,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return jp.array([
        -u[0] * v[1] - u[1] * v[2] - u[2] * v[3],
        u[0] * v[0] + u[1] * v[3] - u[2] * v[2],
        -u[0] * v[3] + u[1] * v[0] + u[2] * v[1],
        u[0] * v[2] - u[1] * v[1] + u[2] * v[0],
    ])


def relative_quat(q1: jp.ndarray, q2: jp.ndarray) -> jp.ndarray:
    """Returns the relative quaternion from q1 to q2."""
    return quat_mul(q2, quat_inv(q1))

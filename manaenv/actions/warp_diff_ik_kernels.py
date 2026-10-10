"""Warp kernels for differential inverse kinematics."""

from __future__ import annotations

import warp as wp

from manaenv.utils.warp_compat import configure_warp_cuda_compilation

configure_warp_cuda_compilation()

# The task normal matrix can be badly conditioned near an arm singularity. Keep
# the fixed-size solve in float64 even though the articulation buffers are
# float32; this avoids the cancellation that made Warp DLS diverge from
# IsaacLab's Torch solve when the end-effector approaches a workspace limit.
_Mat66d = wp.types.matrix((6, 6), dtype=wp.float64)
_Vec6d = wp.types.vector(length=6, dtype=wp.float64)

IK_METHOD_PINV = 0
IK_METHOD_DLS = 1
IK_METHOD_TRANSPOSE = 2


@wp.func
def _frame_pose(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    env_id: int,
    body_idx: int,
    offset: wp.transformf,
) -> wp.transformf:
    root_t = wp.transformf(root_pos_w[env_id], root_quat_w[env_id])
    body_t = wp.transformf(body_pos_w[env_id, body_idx], body_quat_w[env_id, body_idx])
    body_in_root = wp.transform_multiply(wp.transform_inverse(root_t), body_t)
    return wp.transform_multiply(body_in_root, offset)  # ty: ignore[invalid-return-type] -- Warp stub generic


@wp.func
def _axis_angle_from_quat(quat: wp.quatf) -> wp.vec3f:
    """Mirror IsaacLab's shortest-path XYZW quaternion conversion."""

    sign = 1.0
    if quat[3] < 0.0:
        sign = -1.0
    q = wp.quatf(sign * quat[0], sign * quat[1], sign * quat[2], sign * quat[3])
    vector = wp.vec3f(q[0], q[1], q[2])
    magnitude = wp.length(vector)
    half_angle = wp.atan2(magnitude, q[3])
    angle = 2.0 * half_angle
    denominator = 0.5 - angle * angle / 48.0
    if wp.abs(angle) > 1.0e-6:
        denominator = wp.sin(half_angle) / angle
    if wp.abs(denominator) <= 1.0e-12:
        return wp.vec3f(0.0, 0.0, 0.0)
    return vector / denominator


@wp.func
def _task_jacobian_value(
    jacobian_w: wp.array4d(dtype=wp.float32),
    env_id: int,
    jacobi_body_idx: int,
    jacobi_joint_idx: int,
    row: int,
    root_quat: wp.quatf,
    offset_pos: wp.vec3f,
    offset_rot: wp.quatf,
    orientation_weight: wp.vec3f,
    pose_task: int,
) -> float:
    linear_w = wp.vec3f(
        jacobian_w[env_id, jacobi_body_idx, 0, jacobi_joint_idx],
        jacobian_w[env_id, jacobi_body_idx, 1, jacobi_joint_idx],
        jacobian_w[env_id, jacobi_body_idx, 2, jacobi_joint_idx],
    )
    angular_w = wp.vec3f(
        jacobian_w[env_id, jacobi_body_idx, 3, jacobi_joint_idx],
        jacobian_w[env_id, jacobi_body_idx, 4, jacobi_joint_idx],
        jacobian_w[env_id, jacobi_body_idx, 5, jacobi_joint_idx],
    )
    linear_b = wp.quat_rotate_inv(root_quat, linear_w)
    angular_b = wp.quat_rotate_inv(root_quat, angular_w)

    # Match DifferentialInverseKinematicsAction._compute_frame_jacobian:
    # Jv += -skew(offset) @ Jw == Jw x offset.
    linear_b = linear_b + wp.cross(angular_b, offset_pos)
    angular_b = wp.quat_rotate(offset_rot, angular_b)

    if row < 3:
        return linear_b[row]
    if pose_task == 0:
        return 0.0
    angular_row = row - 3
    return angular_b[angular_row] * orientation_weight[angular_row]


@wp.func
def _solve_linear_system_6(matrix: _Mat66d, rhs: _Vec6d) -> _Vec6d:
    """Solve a positive-definite 6x6 system with pivoted elimination."""

    for pivot_col in range(6):
        pivot_row = pivot_col
        pivot_magnitude = wp.abs(matrix[pivot_col, pivot_col])
        for candidate in range(6):
            if candidate > pivot_col:
                candidate_magnitude = wp.abs(matrix[candidate, pivot_col])
                if candidate_magnitude > pivot_magnitude:
                    pivot_row = candidate
                    pivot_magnitude = candidate_magnitude

        if pivot_row != pivot_col:
            for col in range(6):
                tmp = matrix[pivot_col, col]
                matrix[pivot_col, col] = matrix[pivot_row, col]
                matrix[pivot_row, col] = tmp
            rhs_tmp = rhs[pivot_col]
            rhs[pivot_col] = rhs[pivot_row]
            rhs[pivot_row] = rhs_tmp

        pivot = matrix[pivot_col, pivot_col]
        if wp.abs(pivot) < wp.float64(1.0e-12):
            pivot = wp.float64(1.0e-12)
            matrix[pivot_col, pivot_col] = pivot
        for row in range(6):
            if row > pivot_col:
                factor = matrix[row, pivot_col] / pivot
                for col in range(6):
                    if col >= pivot_col:
                        matrix[row, col] = matrix[row, col] - factor * matrix[pivot_col, col]
                rhs[row] = rhs[row] - factor * rhs[pivot_col]

    result = _Vec6d(wp.float64(0.0))
    for reverse_index in range(6):
        row = 5 - reverse_index
        value = rhs[row]
        for col in range(6):
            if col > row:
                value = value - matrix[row, col] * result[col]
        diagonal = matrix[row, row]
        if wp.abs(diagonal) < wp.float64(1.0e-12):
            diagonal = wp.float64(1.0e-12)
        result[row] = value / diagonal
    return result


@wp.func
def _solve_symmetric_pseudoinverse_6(matrix: _Mat66d, rhs: _Vec6d) -> _Vec6d:
    """Apply a Moore-Penrose solve to a symmetric PSD 6x6 matrix."""

    eigenvectors = _Mat66d(wp.float64(0.0))
    for diagonal in range(6):
        eigenvectors[diagonal, diagonal] = wp.float64(1.0)

    # Fixed-size Jacobi sweeps are deterministic and CUDA-graph safe. Sixteen
    # sweeps are sufficient for the small symmetric task matrix in float32.
    sweep = wp.int32(0)
    while sweep < 16:
        p = wp.int32(0)
        while p < 6:
            q = wp.int32(0)
            while q < 6:
                if q > p:
                    off_diagonal = matrix[p, q]
                    if wp.abs(off_diagonal) > wp.float64(1.0e-12):
                        tau = (matrix[q, q] - matrix[p, p]) / (wp.float64(2.0) * off_diagonal)
                        tangent_sign = wp.float64(1.0)
                        if tau < wp.float64(0.0):
                            tangent_sign = wp.float64(-1.0)
                        tangent = tangent_sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                        cosine = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + tangent * tangent)
                        sine = tangent * cosine

                        app = matrix[p, p]
                        aqq = matrix[q, q]
                        for k in range(6):
                            if k != p and k != q:
                                akp = matrix[k, p]
                                akq = matrix[k, q]
                                new_kp = cosine * akp - sine * akq
                                new_kq = sine * akp + cosine * akq
                                matrix[k, p] = new_kp
                                matrix[p, k] = new_kp
                                matrix[k, q] = new_kq
                                matrix[q, k] = new_kq
                        matrix[p, p] = (
                            cosine * cosine * app - wp.float64(2.0) * sine * cosine * off_diagonal + sine * sine * aqq
                        )
                        matrix[q, q] = (
                            sine * sine * app + wp.float64(2.0) * sine * cosine * off_diagonal + cosine * cosine * aqq
                        )
                        matrix[p, q] = wp.float64(0.0)
                        matrix[q, p] = wp.float64(0.0)

                        for k in range(6):
                            vkp = eigenvectors[k, p]
                            vkq = eigenvectors[k, q]
                            eigenvectors[k, p] = cosine * vkp - sine * vkq
                            eigenvectors[k, q] = sine * vkp + cosine * vkq
                q = q + 1
            p = p + 1
        sweep = sweep + 1

    max_eigenvalue = wp.float64(0.0)
    for index in range(6):
        max_eigenvalue = wp.max(max_eigenvalue, wp.abs(matrix[index, index]))
    cutoff = wp.max(wp.float64(1.0e-12), max_eigenvalue * wp.float64(1.0e-12))

    projected = _Vec6d(wp.float64(0.0))
    for eigen_index in range(6):
        component = wp.float64(0.0)
        for row in range(6):
            component = component + eigenvectors[row, eigen_index] * rhs[row]
        eigenvalue = matrix[eigen_index, eigen_index]
        if eigenvalue > cutoff:
            projected[eigen_index] = component / eigenvalue

    result = _Vec6d(wp.float64(0.0))
    for row in range(6):
        for eigen_index in range(6):
            result[row] = result[row] + eigenvectors[row, eigen_index] * projected[eigen_index]
    return result


@wp.kernel(enable_backward=False)
def compute_diff_ik_target_kernel(
    raw_action: wp.array2d(dtype=wp.float32),
    scale: wp.array2d(dtype=wp.float32),
    clip: wp.array3d(dtype=wp.float32),
    has_clip: int,
    action_dim: int,
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_idx: int,
    offset: wp.transformf,
    command_code: int,
    use_relative: int,
    processed_action: wp.array2d(dtype=wp.float32),
    target_pos: wp.array(dtype=wp.vec3f),
    target_quat: wp.array(dtype=wp.quatf),
):
    """Scale an action and compute its desired root-frame pose."""

    env_id = wp.tid()
    for action_index in range(7):
        if action_index < action_dim:
            value = raw_action[env_id, action_index] * scale[env_id, action_index]
            if has_clip == 1:
                value = wp.clamp(value, clip[env_id, action_index, 0], clip[env_id, action_index, 1])
            processed_action[env_id, action_index] = value

    ee_t = _frame_pose(body_pos_w, body_quat_w, root_pos_w, root_quat_w, env_id, body_idx, offset)
    ee_pos = wp.transform_get_translation(ee_t)
    ee_quat = wp.transform_get_rotation(ee_t)

    desired_pos = ee_pos
    desired_quat = ee_quat
    command_pos = wp.vec3f(
        processed_action[env_id, 0],
        processed_action[env_id, 1],
        processed_action[env_id, 2],
    )
    if command_code == 0:  # position
        desired_pos = ee_pos + command_pos if use_relative == 1 else command_pos
    elif use_relative == 1:  # relative pose
        desired_pos = ee_pos + command_pos
        rotation_vector = wp.vec3f(
            processed_action[env_id, 3],
            processed_action[env_id, 4],
            processed_action[env_id, 5],
        )
        angle = wp.length(rotation_vector)
        delta_quat = wp.quat_identity(dtype=wp.float32)
        if angle > 1.0e-6:
            delta_quat = wp.quat_from_axis_angle(rotation_vector / angle, angle)
        desired_quat = delta_quat * ee_quat
    else:  # absolute pose
        desired_pos = command_pos
        quat = wp.quatf(
            processed_action[env_id, 3],
            processed_action[env_id, 4],
            processed_action[env_id, 5],
            processed_action[env_id, 6],
        )
        quat_norm = wp.sqrt(wp.dot(quat, quat))
        if quat_norm > 1.0e-12:
            desired_quat = quat / quat_norm

    target_pos[env_id] = desired_pos
    target_quat[env_id] = desired_quat


@wp.kernel(enable_backward=False)
def solve_diff_ik_kernel(
    jacobian_w: wp.array4d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    target_pos: wp.array(dtype=wp.vec3f),
    target_quat: wp.array(dtype=wp.quatf),
    joint_ids: wp.array(dtype=wp.int32),
    jacobi_joint_ids: wp.array(dtype=wp.int32),
    body_idx: int,
    jacobi_body_idx: int,
    offset: wp.transformf,
    orientation_weight: wp.vec3f,
    pose_task: int,
    ik_method: int,
    method_parameter: float,
    joint_pos_des: wp.array2d(dtype=wp.float32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
):
    """Compute one environment's DiffIK update and write joint targets."""

    env_id = wp.tid()
    ee_t = _frame_pose(body_pos_w, body_quat_w, root_pos_w, root_quat_w, env_id, body_idx, offset)
    ee_pos = wp.transform_get_translation(ee_t)
    ee_quat = wp.transform_get_rotation(ee_t)
    if wp.dot(ee_quat, ee_quat) <= 1.0e-12:
        for joint_index in range(joint_ids.shape[0]):
            asset_joint_index = joint_ids[joint_index]
            current = joint_pos[env_id, asset_joint_index]
            joint_pos_des[env_id, joint_index] = current
            joint_pos_target[env_id, asset_joint_index] = current
        return

    error = _Vec6d(wp.float64(0.0))
    position_error = target_pos[env_id] - ee_pos
    error[0] = wp.float64(position_error[0])
    error[1] = wp.float64(position_error[1])
    error[2] = wp.float64(position_error[2])
    task_dim = 3
    if pose_task == 1:
        task_dim = 6
        quaternion_error = target_quat[env_id] * wp.quat_inverse(ee_quat)
        orientation_error = _axis_angle_from_quat(quaternion_error)
        error[3] = wp.float64(orientation_error[0] * orientation_weight[0])
        error[4] = wp.float64(orientation_error[1] * orientation_weight[1])
        error[5] = wp.float64(orientation_error[2] * orientation_weight[2])

    task_solution = _Vec6d(wp.float64(0.0))
    if ik_method == IK_METHOD_TRANSPOSE:
        for row in range(6):
            if row < task_dim:
                task_solution[row] = wp.float64(method_parameter) * error[row]
    else:
        normal_matrix = _Mat66d(wp.float64(0.0))
        for row in range(6):
            for col in range(6):
                if row < task_dim and col < task_dim:
                    value = wp.float64(0.0)
                    for joint_index in range(joint_ids.shape[0]):
                        jacobi_joint_index = jacobi_joint_ids[joint_index]
                        row_value = _task_jacobian_value(
                            jacobian_w,
                            env_id,
                            jacobi_body_idx,
                            jacobi_joint_index,
                            row,
                            root_quat_w[env_id],
                            wp.transform_get_translation(offset),
                            wp.transform_get_rotation(offset),
                            orientation_weight,
                            pose_task,
                        )
                        col_value = _task_jacobian_value(
                            jacobian_w,
                            env_id,
                            jacobi_body_idx,
                            jacobi_joint_index,
                            col,
                            root_quat_w[env_id],
                            wp.transform_get_translation(offset),
                            wp.transform_get_rotation(offset),
                            orientation_weight,
                            pose_task,
                        )
                        value = value + wp.float64(row_value) * wp.float64(col_value)
                    normal_matrix[row, col] = value
                elif row == col:
                    normal_matrix[row, col] = wp.float64(1.0)

        if ik_method == IK_METHOD_DLS:
            damping_squared = wp.float64(method_parameter) * wp.float64(method_parameter)
            for diagonal in range(6):
                if diagonal < task_dim:
                    normal_matrix[diagonal, diagonal] = normal_matrix[diagonal, diagonal] + damping_squared
            task_solution = _solve_linear_system_6(normal_matrix, error)
        else:
            task_solution = _solve_symmetric_pseudoinverse_6(normal_matrix, error)
            for row in range(6):
                if row < task_dim:
                    task_solution[row] = wp.float64(method_parameter) * task_solution[row]

    for joint_index in range(joint_ids.shape[0]):
        delta = wp.float64(0.0)
        jacobi_joint_index = jacobi_joint_ids[joint_index]
        for row in range(6):
            if row < task_dim:
                jacobian_value = _task_jacobian_value(
                    jacobian_w,
                    env_id,
                    jacobi_body_idx,
                    jacobi_joint_index,
                    row,
                    root_quat_w[env_id],
                    wp.transform_get_translation(offset),
                    wp.transform_get_rotation(offset),
                    orientation_weight,
                    pose_task,
                )
                delta = delta + wp.float64(jacobian_value) * task_solution[row]
        asset_joint_index = joint_ids[joint_index]
        desired = wp.float64(joint_pos[env_id, asset_joint_index]) + delta
        joint_pos_des[env_id, joint_index] = wp.float32(desired)
        joint_pos_target[env_id, asset_joint_index] = wp.float32(desired)


@wp.kernel(enable_backward=False)
def write_diff_ik_target_kernel(
    joint_pos_des: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    joint_pos_target: wp.array2d(dtype=wp.float32),
):
    """Rewrite cached partial targets without index-based host dispatch."""

    env_id, joint_index = wp.tid()
    joint_pos_target[env_id, joint_ids[joint_index]] = joint_pos_des[env_id, joint_index]


__all__ = [
    "IK_METHOD_DLS",
    "IK_METHOD_PINV",
    "IK_METHOD_TRANSPOSE",
    "compute_diff_ik_target_kernel",
    "solve_diff_ik_kernel",
    "write_diff_ik_target_kernel",
]

import torch
import math
from typing import Tuple

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3-vector or batch of 3-vectors to skew-symmetric matrices.
    Args:
        v (torch.Tensor): Input tensor of shape (..., 3).
    Returns:
        torch.Tensor: Skew-symmetric matrices of shape (..., 3, 3).
    """
    ss = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    ss[..., 0, 1] = -v[..., 2]
    ss[..., 0, 2] = v[..., 1]
    ss[..., 1, 0] = v[..., 2]
    ss[..., 1, 2] = -v[..., 0]
    ss[..., 2, 0] = -v[..., 1]
    ss[..., 2, 1] = v[..., 0]
    return ss

def se3_exp_map(delta: torch.Tensor) -> torch.Tensor:
    """
    SE(3) exponential map.
    Args:
        delta (torch.Tensor): Tangent space vector of shape (..., 6) representing (translation_x, y, z, rotation_x, y, z).
    Returns:
        torch.Tensor: Transformation matrix (or batch of matrices) of shape (..., 4, 4).
    """
    if delta.ndim == 1: # Ensure batch dimension for processing
        delta = delta.unsqueeze(0)

    trans = delta[..., :3]  # (B, 3)
    omega = delta[..., 3:]  # (B, 3)

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True) # (B, 1) keepdim for broadcasting
    theta_sq = theta**2 # (B,1)
    
    # For broadcasting with (B,3,3) matrices later
    theta_b = theta.unsqueeze(-1) # (B,1,1)
    theta_sq_b = theta_sq.unsqueeze(-1) # (B,1,1)

    omega_skew = skew_symmetric(omega) # (B, 3, 3)
    omega_skew_sq = torch.matmul(omega_skew, omega_skew) # (B,3,3)
    
    R = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(delta.shape[0], 3, 3).clone()
    V = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(delta.shape[0], 3, 3).clone()

    # Mask for near zero theta.
    near_zero_mask = theta.squeeze(-1) < 1e-12 # (B,)

    # Indices for applying different formulas
    non_zero_indices = torch.where(~near_zero_mask)[0]
    zero_indices = torch.where(near_zero_mask)[0]
    
    eps = 1e-12 # Small epsilon for numerical stability in divisions

    # Non-zero theta calculations (Rodrigues' formula for R and V)
    if non_zero_indices.numel() > 0:
        theta_nz = theta_b[non_zero_indices]              # (N,1,1)
        theta_sq_nz = theta_sq_b[non_zero_indices]        # (N,1,1)
        omega_skew_nz = omega_skew[non_zero_indices]    # (N,3,3)
        omega_skew_sq_nz = omega_skew_sq[non_zero_indices] # (N,3,3)

        sin_theta_nz = torch.sin(theta_nz)
        cos_theta_nz = torch.cos(theta_nz)
        
        # R = I + (sin(theta)/theta)*omega_skew + ((1-cos(theta))/theta^2)*omega_skew_sq
        R_update = (sin_theta_nz / (theta_nz + eps)) * omega_skew_nz + \
                   ((1 - cos_theta_nz) / (theta_sq_nz + eps)) * omega_skew_sq_nz
        R[non_zero_indices] += R_update

        # V = I + ((1-cos(theta))/theta^2)*omega_skew + ((theta-sin(theta))/theta^3)*omega_skew_sq
        V_update = ((1 - cos_theta_nz) / (theta_sq_nz + eps)) * omega_skew_nz + \
                   ((theta_nz - sin_theta_nz) / (theta_nz * theta_sq_nz + eps)) * omega_skew_sq_nz
        V[non_zero_indices] += V_update
        
    # Near zero theta (use Taylor expansion)
    if zero_indices.numel() > 0:
        omega_skew_z = omega_skew[zero_indices]         # (Z,3,3)
        omega_skew_sq_z = omega_skew_sq[zero_indices]   # (Z,3,3)
        
        # R approx I + omega_skew + 0.5 * omega_skew_sq
        R_update_z = omega_skew_z + 0.5 * omega_skew_sq_z
        R[zero_indices] += R_update_z
        
        # V approx I + 0.5 * omega_skew + 1/6 * omega_skew_sq
        V_update_z = 0.5 * omega_skew_z + (1.0/6.0) * omega_skew_sq_z
        V[zero_indices] += V_update_z

    t = torch.matmul(V, trans.unsqueeze(-1)).squeeze(-1) # (B,3)

    T_output = torch.eye(4, device=delta.device, dtype=delta.dtype).expand(delta.shape[0], 4, 4).clone()
    T_output[..., :3, :3] = R
    T_output[..., :3, 3] = t
    
    return T_output.squeeze(0) if delta.shape[0] == 1 and T_output.ndim > 2 and delta.ndim == 1 else T_output


def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    """
    SE(3) logarithm map.
    T: (..., 4, 4) transformation matrix
    Returns: (..., 6) tensor (translation_x, y, z, rotation_x, y, z)
    """
    input_was_unbatched = T.ndim == 2
    if input_was_unbatched: # Ensure batch dimension for processing
        T = T.unsqueeze(0)

    R = T[..., :3, :3] # (B,3,3)
    t = T[..., :3, 3]  # (B,3)

    # trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] # (B,)
    trace = torch.diagonal(R, offset=0, dim1=-2, dim2=-1).sum(-1) # (B,)
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-12, 1.0 - 1e-12) # (B,)
    theta = torch.acos(cos_theta) # (B,)
    
    omega = torch.zeros_like(t) # (B, 3)
    V_inv = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R).clone() # (B,3,3)

    # Masks
    near_zero_mask = theta < 1e-12 # (B,)
    valid_mask = ~near_zero_mask # (B,) (Handle non-zero theta; pi case might still use this formula with care)
    
    non_zero_indices = torch.where(valid_mask)[0]
    if non_zero_indices.numel() > 0:
        theta_v = theta[non_zero_indices].unsqueeze(-1).unsqueeze(-1) # (N,1,1) for broadcasting
        R_v = R[non_zero_indices] # (N,3,3)
        cos_theta_v = cos_theta[non_zero_indices].unsqueeze(-1).unsqueeze(-1) # (N,1,1)

        omega_unscaled_v = torch.zeros_like(t[non_zero_indices]) # (N,3)
        omega_unscaled_v[..., 0] = R_v[..., 2, 1] - R_v[..., 1, 2]
        omega_unscaled_v[..., 1] = R_v[..., 0, 2] - R_v[..., 2, 0]
        omega_unscaled_v[..., 2] = R_v[..., 1, 0] - R_v[..., 0, 1]
        
        # sin_theta from sqrt(1-cos^2(theta)) for stability
        sin_theta_v_sq = torch.clamp(1.0 - cos_theta_v**2, min=0.0) # (N,1,1)
        sin_theta_v = torch.sqrt(sin_theta_v_sq) # (N,1,1)
        
        # --- Direct axis-angle extraction (robust also near θ≈π) ---
        # We know R - R^T = 2 sinθ [a]_x  ⇒  vee(R-R^T) = 2 sinθ · a
        # so a = vee(R-R^T)/(2 sinθ), and rotation vector ω = θ·a
        eps = 1e-12
        vee_R = omega_unscaled_v                                   # (N,3)
        # sin_theta_v is (N,1,1) → squeeze to (N,1) for division
        sin_v = sin_theta_v.squeeze(-1)                            # (N,1)
        axis = vee_R / (2 * sin_v + eps)                           # (N,3)
        theta_vec = theta_v.squeeze(-1)                            # (N,1)
        omega[non_zero_indices] = axis * theta_vec                 # (N,3)
        
        omega_skew_v = skew_symmetric(omega[non_zero_indices]) # (N,3,3)
        omega_skew_sq_v = torch.matmul(omega_skew_v, omega_skew_v) # (N,3,3)
        
        # --- unified-shape beta(theta) for the V_inv update  -----------------
        eps = 1e-12
        # Taylor series for beta: 1/12 - theta^2/720, shape (N,1,1)
        beta_taylor = (1.0/12.0) - (theta_v**2) / 720.0
        # Standard formula: (1/theta^2) - ((1+cosθ)/(2 θ sinθ)), shape (N,1,1)
        denom = 2 * theta_v * sin_theta_v
        beta_std = (1.0 / (theta_v**2 + eps)) - ((1 + cos_theta_v) / (denom + eps))
        # Mask for small denom, same shape (N,1,1)
        mask = denom.abs() < eps
        # Select Taylor vs standard → (N,1,1)
        beta_val = torch.where(mask, beta_taylor, beta_std)
        # Now this (N,1,1) will broadcast cleanly over your (N,3,3) skew matrices
        V_inv_update = -0.5 * omega_skew_v + beta_val * omega_skew_sq_v
        V_inv[non_zero_indices] += V_inv_update

    # near_zero_mask: omega is already zero (correct), V_inv is Identity (correct as omega_skew_z would be 0).
    trans = torch.matmul(V_inv, t.unsqueeze(-1)).squeeze(-1) # (B,3)
    log_map_output = torch.cat((trans, omega), dim=-1) # (B,6)

    return log_map_output.squeeze(0) if input_was_unbatched else log_map_output

def quaternion_to_rotation_matrix(
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Convert quaternion(s) [w, x, y, z] to rotation matrix(ices).

    Args:
        q: Tensor of shape (..., 4) representing quaternion(s) [w, x, y, z].
    Returns:
        Tensor of shape (..., 3, 3) corresponding rotation matrix(ices).
    """
    # Normalize quaternion
    q_norm = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q_norm.unbind(dim=-1)
    # Precompute terms
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    # Construct rotation matrix elements
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz
    return torch.stack(
        [m00, m01, m02, m10, m11, m12, m20, m21, m22],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

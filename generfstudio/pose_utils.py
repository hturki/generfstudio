import numpy as np
import torch
import torch.nn.functional as F


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(
        lookdir, up, position
):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path(
        poses,
        n_frames=120,
        const_speed=True,
        z_variation=0.0,
        z_phase=0.0,
        scale_factor = 1,
):
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * scale_factor
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                (np.cos(theta * 2) * 0.5 + 0.5 + 0.5) * (low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5)) +
                (1 - (np.cos(theta * 2) * 0.5 + 0.5 - 0.5)) * center[0],
                (np.cos(theta * 2) * 0.5 + 0.5 + 0.5) * (low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5)) + (
                        1 - (np.cos(theta * 2) * 0.5 + 0.5 - 0.5)) * center[1],

                # (
                #     offset[0]
                #     + (high - low)[0] * z_variation
                #     * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                # ),
                # (
                #     offset[1] * z_variation
                #     + (high - low)[1] * z_variation
                #     * (np.sin(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                # ),
                z_variation
                * (
                        z_low[2]
                        + (z_high - z_low)[2]
                        * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample(theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def sample(
        t,
        w_logits,
        num_samples,
        deterministic_center=False,
):
    """Piecewise-Constant PDF sampling from a step function.
  
    Args:
      rng: random number generator (or None for `linspace` sampling).
      t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
      w_logits: [..., num_bins], logits corresponding to bin weights
      num_samples: int, the number of samples.
      single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
      deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
  
    Returns:
      t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    if deterministic_center:
        pad = 1 / (2 * num_samples)
        u = np.linspace(pad, 1.0 - pad - eps, num_samples)
    else:
        u = np.linspace(0, 1.0 - eps, num_samples)
    u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))

    return invert_cdf(u, t, w_logits)


def integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
  
    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.
  
    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.
  
    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = np.minimum(1, np.cumsum(w[Ellipsis, :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw, np.ones(shape)], axis=-1)
    return cw0


def invert_cdf(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = F.softmax(torch.from_numpy(w_logits), dim=-1).numpy()
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = sorted_interp(u, cw, t)
    return t_new


def sorted_interp(
        x, xp, fp, eps=np.finfo(np.float32).eps ** 2
):
    """A version of interp() where xp and fp must be sorted."""
    (xp0, xp1), (fp0, fp1) = sorted_lookup(
        x, xp, (xp, fp)
    )[1]
    offset = np.clip((x - xp0) / np.maximum(eps, xp1 - xp0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


def sorted_lookup(x, xp, fps):
    """Lookup `x` into locations `xp` , return indices and each `[fp]` value."""
    if not isinstance(fps, tuple):
        raise ValueError(f'Input `fps` must be a tuple, but is {type(fps)}.')

    # jnp.searchsorted() has slightly different conventions for boundary
    # handling than the rest of this codebase.
    # idx = jax.vmap(lambda a, v: np.searchsorted(a, v, side='right'))(
    #     xp.reshape([-1, xp.shape[-1]]), x.reshape([-1, x.shape[-1]])
    # ).reshape(x.shape)
    idx = np.array(
        [np.searchsorted(a, v, side='right') for a, v in zip(xp.reshape([-1, xp.shape[-1]]), x.reshape([-1, x.shape[-1]]))]
    ).reshape(x.shape)
    idx1 = np.minimum(idx, xp.shape[-1] - 1)
    idx0 = np.maximum(idx - 1, 0)
    vals = []
    for fp in fps:
        fp0 = np.take_along_axis(fp, idx0, axis=-1)
        fp1 = np.take_along_axis(fp, idx1, axis=-1)
        vals.append((fp0, fp1))

    return (idx0, idx1), vals

import torch
from optimization.quad_scenario import dt, mass, drag_coefficient, u_max, obs_positions, obs_radii


def unnormalize(data_norm, mean, std):
    return data_norm * (std + 1e-6) + mean


def quad_dynamics_batch(x, u):
    """Compute state derivative for batch inputs."""
    v = x[:, 3:]
    accel = u / mass - drag_coefficient * torch.norm(v, dim=1, keepdim=True) * v / mass
    return torch.cat((v, accel), dim=1)


def rollout_with_actions(initial_state, actions):
    """Propagate dynamics using Euler integration."""
    batch_size, horizon, _ = actions.shape
    state = initial_state
    states = []
    for t in range(horizon):
        state = state + quad_dynamics_batch(state, actions[:, t, :]) * dt
        states.append(state)
    return torch.stack(states, dim=1)


def physics_based_loss(states_norm, action_preds_norm, data_stats):
    device = states_norm.device
    states_mean = data_stats['states_mean'].to(device)
    states_std = data_stats['states_std'].to(device)
    actions_mean = data_stats['actions_mean'].to(device)
    actions_std = data_stats['actions_std'].to(device)

    # unnormalize sequences
    states = unnormalize(states_norm, states_mean, states_std)
    actions = unnormalize(action_preds_norm, actions_mean, actions_std)

    # propagate using actions except last one
    pred_states = rollout_with_actions(states[:, 0, :], actions[:, :-1, :])

    true_states = states[:, 1:, :]
    mse_loss = torch.mean((pred_states - true_states) ** 2)

    # Constraint penalties
    # Action bounds
    action_violation = torch.clamp(torch.abs(actions) - u_max, min=0.0)
    ctrl_penalty = torch.mean(action_violation ** 2)

    # Keep-out-zone (obstacle) violation
    if obs_positions is not None:
        obs_pos = torch.tensor(obs_positions, device=device)
        obs_rad = torch.tensor(obs_radii, device=device)
        dist = torch.norm(pred_states[:, :, None, :3] - obs_pos, dim=3) - obs_rad
        Koz_violation = torch.clamp(-dist, min=0.0)
        koz_penalty = torch.mean(Koz_violation ** 2)
    else:
        koz_penalty = torch.tensor(0.0, device=device)

    # Weight penalties higher than state mismatch
    return mse_loss + 10.0 * koz_penalty + ctrl_penalty

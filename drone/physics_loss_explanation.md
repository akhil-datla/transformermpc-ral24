# Physics Based Loss for Drone Model

The original training scripts minimized the mean squared error (MSE) between the predicted states/actions and the values stored in the dataset. This commit introduces a physics based loss exploiting the quadrotor dynamics.

## Implementation

A new module `drone/dynamics/losses.py` defines utilities:

- **`quad_dynamics_batch`** – computes the state derivative for a batch of states and control inputs using the point–mass model from `quad_scenario`.
- **`rollout_with_actions`** – performs Euler integration of the dynamics for a sequence of actions.
- **`physics_based_loss`** – unnormalizes states and predicted actions, propagates the initial state with the dynamics and computes the MSE between the propagated trajectory and the true states.

Training scripts (`main_train.py` and `dagger_training.py`) import `physics_based_loss` and replace the previous state loss with this new term. Actions are still penalized with a standard MSE. Dataset statistics are retrieved from the training loader to allow unnormalization.

## Usage

During training or evaluation the loss is computed as

```python
loss_i_action = torch.mean((action_preds - actions_i) ** 2)
loss_i_state = physics_based_loss(states_i, action_preds, data_stats)
loss = loss_i_action + loss_i_state
```

where `states_i` and `action_preds` contain normalized values and `data_stats` stores their means and standard deviations.

This loss encourages the network to output control sequences that are physically consistent with the quadrotor model rather than only matching the next observed state.

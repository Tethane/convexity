import numpy as np
import pytest
from convexity.math import control, reinforcement as rl


def model():
    return control.FiniteMDP([[[1.0], [1.0]]], [[0.0, 1.0]], 0.5)


def test_q_learning_and_fitted_q_recover_exact_control():
    mdp = model()
    exact = control.value_iteration(mdp)
    sampled = rl.q_learning(mdp, steps=30000, epsilon=0.4, rng=np.random.default_rng(3))
    assert sampled.policy[0] == exact.policy[0] == 1
    assert sampled.q_values[0, 1] == pytest.approx(2.0, abs=0.02)
    fitted = rl.fitted_q(mdp, np.eye(2).reshape(1, 2, 2))
    assert fitted.residual < 1e-8
    np.testing.assert_allclose(fitted.q_values, [[1.0, 2.0]], atol=1e-8)


def test_targets_terminal_masks_and_loss():
    targets = rl.bellman_targets(
        [1, 2],
        [[3, 100], [99, 99]],
        [False, True],
        feasible=[[True, False], [False, False]],
        discount=0.5,
    )
    np.testing.assert_allclose(targets, [2.5, 2.0])
    assert rl.bellman_loss([1, 2], [2, 2], weights=[0.25, 0.75]) == 0.25
    with pytest.raises(ValueError):
        rl.bellman_targets([0], [[1, 2]], [False], feasible=[[False, False]])


def test_tensor_actor_critic_gradients_and_hard_masks():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, -2.0]], requires_grad=True)
    values = torch.tensor([0.0, 0.0], requires_grad=True)
    returns = torch.tensor([1.0, -1.0], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = rl.actor_critic_loss(
        logits, torch.tensor([0, 1]), values, returns, mask, entropy_weight=0.01
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, 2] == 0).all()
    assert logits.grad[0, 0] < 0 and logits.grad[1, 1] > 0
    assert returns.grad is None
    assert values.grad is not None
    with pytest.raises(ValueError):
        rl.actor_critic_loss(logits, torch.tensor([2, 1]), values, returns, mask)


@pytest.mark.parametrize("algorithm", ["a2c", "ppo"])
def test_real_stable_baselines_backend(algorithm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        agent = rl.train_agent(
            model(),
            algorithm=algorithm,
            steps=512,
            seed=11,
            n_steps=32,
            learning_rate=0.01,
        )
        assert agent.action(0) == 1
    finally:
        torch.set_num_threads(previous_threads)
    masked = control.FiniteMDP(
        [[[1.0], [1.0]]], [[0.0, 1.0]], 0.5, feasible=[[True, False]]
    )
    with pytest.raises(ValueError):
        rl.train_agent(masked, steps=10)

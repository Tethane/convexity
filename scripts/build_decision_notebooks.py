"""Build and execute generated-data notebooks for families 27–36."""

import argparse
from pathlib import Path
import sys
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """
from pathlib import Path
import sys
root = Path.cwd()
if not (root / 'convexity').exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
plt.rcParams.update({'figure.figsize': (9, 3.5), 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .2})
print('Interpreter:', sys.executable)
"""

BOOKS = {
    "risk": (
        "Weighted tails and whole-path risk",
        "Losses are adverse when positive. ES integrates exactly the worst tail mass, including a fractional atom at VaR. Path breaches refer only to supplied observation times.",
        "How atoms change VaR and ES",
        "A finite law has stepwise VaR. ES averages exactly the remaining tail mass and stays continuous in the confidence level below one. The two paths have different maximum drawdowns despite ending above their interim lows.",
        """
levels = np.linspace(0, .99, 150)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(levels, [risk.value_at_risk(losses, a, weights=weights) for a in levels], label='VaR')
axes[0].plot(levels, [risk.expected_shortfall(losses, a, weights=weights) for a in levels], label='ES')
axes[0].set(xlabel='Confidence level', ylabel='Loss'); axes[0].legend()
axes[1].plot(paths.times, paths.values.T, marker='o')
axes[1].axhline(92., color='red', linestyle='--', label='Breach threshold')
axes[1].set(xlabel='Time', ylabel='Wealth'); axes[1].legend(); plt.show()
""",
    ),
    "utility": (
        "Utility over positive and negative wealth",
        "The options-sheet utility is increasing and concave on all real wealth. Log and CRRA utility instead require positive wealth. Certainty equivalents are wealth units; expected utility generally is not.",
        "Risk aversion and certainty equivalents",
        "The same wealth law produces lower certainty equivalents as aversion increases. Noncash switching preferences are additional decision criteria and never reduce account cash.",
        """
x = np.linspace(-20, 20, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for aversion in (.1, .5, .9):
    model = utility.SmoothUtility(aversion, 3.)
    axes[0].plot(x, model(x), label=f'lambda={aversion}')
a = np.linspace(.01, .99, 80)
axes[1].plot(a, [utility.certainty_equivalent(wealth, utility.SmoothUtility(z, 3.)) for z in a])
axes[1].axhline(wealth.mean(), color='grey', linestyle='--', label='Mean wealth')
axes[0].set(xlabel='Wealth', ylabel='Utility'); axes[0].legend()
axes[1].set(xlabel='Aversion', ylabel='Certainty equivalent'); axes[1].legend(); plt.show()
""",
    ),
    "optimization": (
        "Optimization solutions and their constraints",
        "QuadraticProgram minimizes one-half x-transpose Q x plus c-transpose x. Bounds are free by default. Solver status and primal feasibility accompany each solution; local nonlinear success is not a global certificate.",
        "A constrained quadratic allocation path",
        "Increasing the variance penalty changes the allocation toward the lower-variance asset. At a bound, the active set changes. The equality constraint fixes total invested weight at one.",
        """
penalties = np.geomspace(.1, 20, 50)
solutions = []
for penalty in penalties:
    p = opt.QuadraticProgram([-.05, -.1], quadratic=penalty*np.diag([.01, .04]),
                              equality=[[1., 1.]], target=[1.], lower_bounds=0.)
    solutions.append(opt.solve(p).solution)
solutions = np.array(solutions)
plt.semilogx(penalties, solutions[:, 0], label='Lower-variance asset')
plt.semilogx(penalties, solutions[:, 1], label='Higher-mean asset')
plt.xlabel('Variance penalty'); plt.ylabel('Allocation'); plt.legend(); plt.show()
display(pd.DataFrame(solutions[::10], columns=['Asset 1', 'Asset 2'], index=penalties[::10]))
""",
    ),
    "robust": (
        "Robust decisions under declared ambiguity",
        "An ellipsoid describes uncertain means, while finite-support KL ambiguity changes scenario probabilities. Neither adds unmodeled tail outcomes. Zero nominal mass stays zero under KL(q||p).",
        "Ambiguity changes allocations and worst-case losses",
        "The second asset has greater mean uncertainty, so its allocation changes with ellipsoid radius. Increasing the KL radius can only increase the worst expected loss, approaching the worst supported outcome.",
        """
radii = np.linspace(0, 2, 40)
allocations = np.array([robust.robust_allocation(mean, covariance,
    robust.Ellipsoid(mean, [[.01, 0], [0, .05]], radius), aversion=2.).solution for radius in radii])
kl_radii = np.linspace(0, 3.1, 35)
worst = [robust.kl_worst_case([0., 2., 10.], radius, weights=[.7, .25, .05]).objective for radius in kl_radii]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(radii, allocations[:, 0], label='Asset 1'); axes[0].plot(radii, allocations[:, 1], label='Asset 2')
axes[0].set(xlabel='Ellipsoid radius', ylabel='Allocation'); axes[0].legend()
axes[1].plot(kl_radii, worst); axes[1].set(xlabel='KL radius', ylabel='Worst expected loss'); plt.show()
""",
    ),
    "control": (
        "Bellman values and decisions before observing shocks",
        "A finite MDP supplies all state, action and transition axes. A scenario tree supplies observed histories. The first action cannot depend on which future branch will occur.",
        "Horizon effects and the value of information",
        "The stopping-like MDP chooses the immediate reward near the endpoint and continuation over a longer horizon. A scenario-specific first action would be clairvoyant: the constrained tree uses one shared root action.",
        """
horizons = np.arange(1, 31)
values = [control.finite_horizon(mdp, int(h)).values[0, 0] for h in horizons]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(horizons, values); axes[0].axhline(10., linestyle='--', color='grey')
axes[0].set(xlabel='Horizon', ylabel='Initial value')
axes[1].bar(['Nonanticipative', 'Clairvoyant lower cost'], [plan.objective, 2.5])
axes[1].set(ylabel='Expected quadratic cost'); plt.show()
observations = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
identity = np.repeat(np.eye(2)[:, None, :], 2, axis=1)
belief_model = control.FiniteMDP(identity, [[1., 0.], [0., 1.]], 1.)
print('Two-step value with future observation:', control.belief_plan(belief_model, observations, [.5, .5], 2).value)
print('Two-step value without information:', control.belief_plan(belief_model, np.ones((2, 2, 1)), [.5, .5], 2).value)
""",
    ),
    "policies": (
        "Hysteresis, closure and no-action bands",
        "The deadline belongs to the immutable state and is never rolled forward. Closed states are absorbing. Holding wins a tie after switching preferences; mandatory feasible actions override waiting.",
        "A fixed lifecycle and a switching boundary",
        "A probability dip inside the hysteresis band retains activation. Falling to the exit threshold closes the record even if confidence later recovers. This lifecycle is distinct from the choice to hold zero exposure.",
        """
times = np.arange(8.)
probabilities = [.5, .91, .8, .7, .6, .99, .99, .99]
s = policies.PolicyState(7.)
modes = []
for t, p in zip(times, probabilities):
    s = rule.update(s, p, t)
    modes.append({'waiting': 0, 'active': 1, 'closed': 2}[s.mode])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(times, probabilities, marker='o'); axes[0].axhline(.9, linestyle='--'); axes[0].axhline(.6, linestyle='--')
axes[0].set(xlabel='Time', ylabel='Probability')
axes[1].step(times, modes, where='post'); axes[1].set(yticks=[0, 1, 2], yticklabels=['Waiting', 'Active', 'Closed'], xlabel='Time')
plt.show()
gains = np.linspace(0, 1, 11)
display(pd.DataFrame({'Alternative advantage': gains,
    'Chosen action': [policies.no_action([5., 5.+g], switching=.5).action for g in gains]}))
""",
    ),
    "execution": (
        "Execution schedules and assignment likelihoods",
        "Liquidation trades reduce signed inventory to zero. Temporary impact scales with trade squared divided by interval length. Inventory risk is a preference and is reported separately from cash cost.",
        "Front-loading inventory and recovering assignment probabilities",
        "Higher inventory aversion accelerates liquidation and generally raises expected cash cost. Assignment probabilities below are recovered from generated independent binomial counts under the declared logistic specification.",
        """
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
rows = []
for aversion in (0., 2., 10.):
    result = execution.liquidate([10.], [.25]*4, spread=.01, impact=[[.5]], covariance=[[1.]], aversion=aversion)
    axes[0].plot(np.linspace(0, 1, 5), result.inventory[:, 0], marker='o', label=f'aversion={aversion}')
    rows.append([aversion, result.expected_cost, result.inventory_penalty])
grid = np.linspace(-3, 3, 200)
features = np.column_stack([np.ones_like(grid), grid])
axes[1].plot(grid, 1/(1+np.exp(-features @ true_beta)), label='Truth')
axes[1].plot(grid, fit.probability(features), linestyle='--', label='Fitted')
axes[0].set(xlabel='Execution time', ylabel='Remaining inventory'); axes[0].legend()
axes[1].set(xlabel='Feature', ylabel='Assignment probability'); axes[1].legend(); plt.show()
display(pd.DataFrame(rows, columns=['Aversion', 'Cash cost', 'Risk preference']))
""",
    ),
    "wealth": (
        "Self-financing cash and delivery identities",
        "Marks are per held asset unit, and option marks already include contract multipliers. Pending delivery is an owned claim. Settlement moves it into settled cash and holdings once without creating wealth.",
        "Conservation through delivery and the Taylor remainder",
        "Exercise at intrinsic value and subsequent settlement conserve marked wealth. Away from the expansion point, delta and gamma leave an explicit repricing remainder.",
        """
a = wealth.Account([2., 0.], 10000.)
b = wealth.exercise_delivery(a, 0, 2, underlying_index=1, shares_per_contract=100, cash_per_contract=-10000)
c = wealth.settle(b)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].bar(['Before', 'Pending', 'Settled'], [z.wealth([1000., 110.]) for z in (a, b, c)])
axes[0].set(ylabel='Marked wealth')
shocks = np.linspace(-1., 1., 100)
results = [wealth.pnl_attribution(lambda x: x[0]**3, [2.], [2.+shock], [12.], [[12.]]) for shock in shocks]
axes[1].plot(shocks, [z.exact for z in results], label='Exact')
axes[1].plot(shocks, [z.first_order+z.second_order for z in results], label='Delta + gamma', linestyle='--')
axes[1].set(xlabel='State change', ylabel='P&L'); axes[1].legend(); plt.show()
""",
    ),
    "diagnostics": (
        "Dependent uncertainty and proper forecast scores",
        "Residual tests state their null hypothesis. Block bootstrap and HAC covariance target dependence-aware uncertainty under stationarity assumptions; neither fixes a misspecified forecast.",
        "Dependence changes mean uncertainty",
        "IID resampling misses serial dependence in this AR sample. Larger blocks preserve more of that dependence, with greater Monte Carlo variability. Scores are evaluated on a supplied probability law, independently of resampling uncertainty.",
        """
blocks = [1, 3, 5, 10, 20, 40]
standard_errors = []
for block in blocks:
    result = diagnostics.block_bootstrap(np.mean, series, block_length=block, draws=400, rng=np.random.default_rng(3500+block))
    standard_errors.append(result.samples.std(ddof=1))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(blocks, standard_errors, marker='o'); axes[0].set(xlabel='Block length', ylabel='Bootstrap mean standard error')
ensemble = np.random.default_rng(3501).normal(size=1000)
shifts = np.linspace(-2, 2, 50)
scores = [diagnostics.crps(ensemble+shift, 0.) for shift in shifts]
axes[1].plot(shifts, scores); axes[1].set(xlabel='Forecast location shift', ylabel='CRPS at observation zero'); plt.show()
""",
    ),
    "reinforcement": (
        "Approximate values against an exact Bellman reference",
        "RL is optional. The finite reference exposes Bellman error directly. A small training residual is evidence about this supplied simulator, not predictive quality or profitability.",
        "Sample budgets and an actor-critic adapter",
        "One-hot fitted Q recovers the finite reference. Sampled Q learning approaches it as transitions accumulate. The optional actor-critic run below uses a real Stable-Baselines3 backend on the same simple model.",
        """
budgets = [100, 500, 2000, 10000, 30000]
residuals = [reinforcement.q_learning(mdp, steps=n, epsilon=.4, rng=np.random.default_rng(3600)).residual for n in budgets]
plt.loglog(budgets, np.maximum(residuals, 1e-16), marker='o', label='Sampled Q')
plt.axhline(fitted.residual, linestyle='--', label='One-hot fitted Q')
plt.xlabel('Sampled transitions'); plt.ylabel('Bellman residual'); plt.legend(); plt.show()
agent = reinforcement.train_agent(mdp, steps=512, seed=11, learning_rate=.01)
print('Actor-critic action:', agent.action(0), 'exact action:', exact.policy[0])
""",
    ),
}


def build(name, index):
    title, intro, heading, narrative, code = BOOKS[name]
    script = "optimal_control" if name == "control" else name
    example = (ROOT / "examples" / f"{script}.py").read_text()
    cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nFamily {index}. {intro}"),
        nbf.v4.new_code_cell(textwrap.dedent(SETUP).strip()),
        nbf.v4.new_markdown_cell(
            "## Public workflow\n\nThe executable example uses only supplied or generated inputs."
            + (
                "\n\nThe standalone script is `examples/optimal_control.py` to avoid shadowing the python-control provider."
                if name == "control"
                else ""
            )
        ),
        nbf.v4.new_code_cell(example),
        nbf.v4.new_markdown_cell(f"## {heading}\n\n{narrative}"),
        nbf.v4.new_code_cell(textwrap.dedent(code).strip()),
        nbf.v4.new_markdown_cell(
            "## Scope of this evidence\n\nThese experiments illustrate the declared finite catalog, not live trading performance. "
            "See the [contracts and limitations](../docs/decision_mathematics.md) and "
            "`tests/math/decisions/` for mathematical acceptance checks."
        ),
    ]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "convexity": {
                "family": index,
                "data": "generated",
                "generator": "scripts/build_decision_notebooks.py",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="*", choices=list(BOOKS))
    args = parser.parse_args()
    for name in args.only or BOOKS:
        notebook = build(name, 27 + list(BOOKS).index(name))
        if args.execute:
            import os
            from nbclient import NotebookClient
            from jupyter_client.manager import KernelManager

            km = KernelManager(kernel_name="python3")
            spec = km.kernel_spec
            if spec is None:
                raise RuntimeError("python3 kernel specification is unavailable")
            spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            client = NotebookClient(
                notebook,
                timeout=300,
                km=km,
                resources={"metadata": {"path": str(ROOT)}},
            )
            try:
                client.execute(
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                )
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
        nbf.validate(notebook)
        nbf.write(notebook, ROOT / "notebooks" / f"{name}.ipynb")
        print(name, "executed" if args.execute else "built", flush=True)


if __name__ == "__main__":
    main()

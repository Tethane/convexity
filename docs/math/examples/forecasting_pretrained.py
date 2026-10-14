"""Real-checkpoint integration verification (downloads cached pretrained weights).

Run explicitly in an environment with convexity[forecasting-advanced].
Artifacts stay in .cache/forecasting. These checks test integration and one-step
fine-tuning, not model quality or investment performance.
"""

import gc
import json
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np

from convexity.math import forecasting as f


def main():
    import torch
    from dataclasses import replace

    torch.set_num_threads(2)
    cache = Path(".cache/forecasting")
    cache.mkdir(parents=True, exist_ok=True)
    data = np.sin(np.arange(96.0) / 6) + np.arange(96.0) / 200
    report_path = cache / "pretrained-verification.json"
    report = (
        json.loads(report_path.read_text())
        if sys.argv[1:] and report_path.exists()
        else {}
    )
    for provider in sys.argv[1:] or (
        "chronos",
        "chronos_bolt",
        "chronos2",
        "timesfm3",
        "timesfm2.5",
    ):
        print("Loading", provider, flush=True)
        options = (
            {"use_continuous_quantile_head": True} if provider == "timesfm2.5" else {}
        )
        key = cast(
            Literal["chronos", "chronos_bolt", "chronos2", "timesfm3", "timesfm2.5"],
            provider,
        )
        if options:
            spec = f.load_pretrained(
                key,
                cache_dir=cache / "hub",
                device="cpu",
                use_continuous_quantile_head=True,
            )
        else:
            spec = f.load_pretrained(key, cache_dir=cache / "hub", device="cpu")
        fitted = spec.fit(data)
        forecast = fitted.forecast(4, rng=np.random.default_rng(3))
        assert forecast.shape == (4, 1) and np.isfinite(forecast.point).all()
        assert forecast.quantile(0.1).shape == (4, 1)
        report[provider] = {
            "checkpoint": spec.checkpoint,
            "shape": list(forecast.shape),
            "point": forecast.point.tolist(),
            "capabilities": sorted(forecast.capabilities),
        }
        if provider == "chronos":
            sampled = replace(spec, output="samples", samples=8).fit(data).forecast(4)
            assert sampled.paths is not None
            assert sampled.paths.shape == (8, 4, 1)
            report[provider]["samples_verified"] = True
        if provider == "chronos2":
            history = f.TimeSeries(
                np.column_stack((data, data * 0.5)),
                covariates=f.Covariates(
                    known_future=np.arange(96.0), observed_past=np.cos(np.arange(96.0))
                ),
            )
            conditional = spec.fit(history).forecast(
                4, covariates=np.arange(96.0, 100.0)
            )
            assert conditional.shape == (4, 2)
            fine = replace(
                spec,
                finetune_steps=1,
                finetune_horizon=4,
                finetune_options={
                    "output_dir": str(cache / "finetuned"),
                    "batch_size": 2,
                    "context_length": 32,
                    "min_past": 4,
                    "save_strategy": "no",
                    "logging_strategy": "no",
                    "disable_tqdm": True,
                },
            ).fit(data)
            assert fine.forecast(4).shape == (4, 1)
            report[provider]["finetuning_verified"] = True
            del fine
        if provider == "timesfm3":
            history = f.TimeSeries(
                np.column_stack((data, data * 0.5)),
                covariates=f.Covariates(
                    known_future=np.arange(96.0), observed_past=np.cos(np.arange(96.0))
                ),
            )
            assert spec.fit(history).forecast(
                4, covariates=np.arange(96.0, 100.0)
            ).shape == (4, 2)
            report[provider]["multivariate_covariates_verified"] = True
        (cache / "pretrained-verification.json").write_text(
            json.dumps(report, indent=2)
        )
        print(provider, "verified", flush=True)
        del fitted, spec, forecast
        gc.collect()
    print(
        "All real-checkpoint integrations passed:",
        cache / "pretrained-verification.json",
    )


if __name__ == "__main__":
    main()

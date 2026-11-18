"""Optuna TPE search with trip-grouped CV and per-fold median pruning.

TPE (Tree-structured Parzen Estimator) models p(params | good trials) and
p(params | bad trials) and proposes the configuration maximising their ratio,
i.e. expected improvement - a sequential model-based (Bayesian) optimiser.
"""
from __future__ import annotations

import time

import numpy as np
import optuna

from ..config import SEED
from ..training.cross_validation import evaluate_model, fit_model, training_rows
from .search_space import normalise, suggest


def make_objective(model_name, cands, obs, folds, stride, extra=None, log=print):
    extra = extra or {}

    def objective(trial: optuna.Trial) -> float:
        params = normalise(model_name, suggest(trial, model_name))
        f1s, top1s, lls = [], [], []
        t0 = time.perf_counter()
        for k, (tr_trips, va_trips) in enumerate(folds):
            tr = training_rows(cands[cands["trip_id"].isin(tr_trips)], stride)
            model, _, _ = fit_model(model_name, params, extra, tr, seed=SEED)
            va_c = cands[cands["trip_id"].isin(va_trips)]
            va_o = obs[obs["trip_id"].isin(va_trips)]
            m, _, _ = evaluate_model(model, va_c, va_o)
            f1s.append(m["macro_f1"]); top1s.append(m["top1_accuracy"]); lls.append(m["log_loss"])
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                trial.set_user_attr("folds_completed", k + 1)
                trial.set_user_attr("duration_s", time.perf_counter() - t0)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1", float(np.mean(top1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_macro_f1_std", float(np.std(f1s)))
        trial.set_user_attr("folds_completed", len(folds))
        trial.set_user_attr("duration_s", time.perf_counter() - t0)
        log(f"  trial {trial.number}: macroF1={np.mean(f1s):.4f} top1={np.mean(top1s):.4f} "
            f"({time.perf_counter() - t0:.0f}s) {params}")
        return float(np.mean(f1s))

    return objective


def run_study(model_name, cands, obs, folds, n_trials, stride, storage=None, seed=SEED, log=print):
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=f"pathsnap_{model_name}", storage=storage, load_if_exists=False)
    study.optimize(make_objective(model_name, cands, obs, folds, stride, log=log), n_trials=n_trials,
                   gc_after_trial=True)
    return study

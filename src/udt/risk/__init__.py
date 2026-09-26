"""udt.risk — module M8 (dev doc §9).

`ensemble.py`: raw ensemble-disagreement statistic (§9.2). `conformal.py`:
split-conformal calibration + interval width (§9.2). `engine.py`: ties
both together with `twin/counterfactual.py`'s `simulate()` output into
one `RiskAssessment` per decision (§9.1), plus `calibrate_from_samples`
for building a `RiskCalibration`. `gate.py`: the autonomy-tier decision
(§9.3), reading thresholds from `configs/autonomy.yaml`. `human_model.py`:
the scripted approve/reject stand-in for headless experiments (§9.3).

See each module's own docstring for the two disclosed prerequisite gaps
every function in this package is honest about (no 5 real trained MAPPO
critics yet; no frozen val/test scenario suite yet) and
`MTP_Module_Planner.md`'s M8 row for the full build story.
"""

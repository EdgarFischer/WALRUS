# Calibration

Calibration estimates simulator parameter distributions from in-vivo data so that generated spectra resemble the target acquisition and population more closely.

## What is calibrated

Depending on the experiment, calibration can include:

- metabolite amplitude distributions;
- zero-order phase;
- frequency shift;
- SNR;
- linewidth-related quantities;
- water and lipid nuisance scaling;
- empirical simulator realism comparisons.

Some paper analyses intentionally exclude parameters that are not consistently defined across simulators or fitting pipelines. For example, water/metabolite ratio, lipid/metabolite ratio, or FWHM may be handled separately when their definitions differ between LCModel output and the simulator.

## Relevant code locations

Calibration and analysis helpers live mainly in:

```text
src/walinet/parameter_calibration/
src/walinet/paper_figures/
scripts/evaluation/
```

Common supporting configs:

```text
configs/Simulation/Metabos_healthy_7T_2sigma.yaml
configs/Simulation/Metabos_healthy_7T_WALINET_Style.yaml
configs/Simulation/Metabos_tumor_7T_2sigma.yaml
configs/PaperFig1/
```

## General workflow

1. Collect LCModel or equivalent fit outputs for selected in-vivo subjects.
2. Extract the parameters of interest into arrays or tables.
3. Apply robust normalization/statistics when comparing distributions, often based on median and IQR.
4. Write calibrated distributions to YAML.
5. Run simulation/evaluation using the calibrated profile.
6. Compare synthetic parameter distributions with held-out in-vivo distributions.

## Robust standardization

Several analyses use robust z-scoring:

```text
z = (x - median_in_vivo) / sigma_robust
sigma_robust = IQR_in_vivo / 1.349
```

This makes distances less sensitive to outliers than mean/standard-deviation scaling.

## Simulator realism analysis

The realism evaluation compares simulated parameter vectors against the in-vivo distribution. A typical procedure is:

1. choose a common set of parameters;
2. robust-standardize each parameter from in-vivo statistics;
3. compute k-nearest-neighbor distances from simulation points to in-vivo points;
4. compare distance distributions between simulator variants.

The relevant script is:

```text
scripts/evaluation/evaluate_7t_simulator_realism.py
```

## Reproducibility notes

- Always record which subjects were used for calibration and which were held out.
- Keep the exact metabolite list fixed between calibrated profiles and comparisons.
- For cyclic phase parameters, use sine/cosine or explicit circular handling rather than treating the angle as an ordinary linear variable.
- Prefer YAML configs over hard-coded constants when a parameter affects training or evaluation.

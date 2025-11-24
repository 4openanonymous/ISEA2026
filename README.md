# Artifact Guide
# ISEA Submission — Audited Artifact Version

This artifact provides the minimal reproducible components required to regenerate the computational results and video renderings referenced in the paper. The aim is not to disclose the full development history but to make verifiable the specific conditions under which the perceptual behaviour of the system can be reconstructed, examined, and audited.

The pipeline contains two reproducible stages:
	1.	Temporal field calibration — translating latent operators $\hat d$ and $\rho$ into duration.
	2.	Perceptual unfolding — letting duration become a visible rhythm through frame allocation.

Both are reproduced entirely from the files and scripts provided here.

⸻

1. Environment

conda env create -f environment.yml
conda activate isea_artifact

This environment contains only the dependencies needed for W2 calibration and W4 rendering.

⸻

2. Required Input Data (Minimal Set)

The artifact includes exactly two CSV inputs from earlier stages of the project.
They form the starting point for all reproducible operations.

2.1 events_2021.csv

Contains the two semantic operators computed in the original W1 stage:
	•	d_hat — local sparsity $\hat d$
	•	rho — neighbourhood contrast $\rho$

These constitute the latent force-field from which temporal dynamics are derived.

⸻

2.2 events_merged_2021__w3-baseline-run.csv

Contains:
	•	dt_applied — the measured physical durations obtained through real-time execution (time.sleep()), capturing CPU jitter and OS latency.

This file enables reproduction of the EMPIRICAL temporal condition.

⸻

3. Reproducing Temporal Calibration (W2)

W2 reconstructs the temporal behaviour of the system by calibrating:
	•	$\epsilon$ — temporal inertia
	•	$\lambda$ — boundary resistance

The baseline configuration (t_min, t_max, $\gamma$) is stored in:

baseline/W2_baseline.json

Run the calibration

python scripts/w2_run_year_wrapper.py \
    --baseline baseline/W2_baseline.json \
    --events data/events_2021.csv \
    --outdir reports_new/2021

Generated Outputs
	•	fi_vs_epsilon.png
	•	rc_vs_lambda.png
	•	W2_perceptual_report_2021.md
	•	selected stable parameters:
	•	$\epsilon^* = 0.20$
	•	$\lambda^* = 0.69$

Compare the new reports with the audited ones:

diff reports/2021/W2_perceptual_report_2021.md \
     reports_new/2021/W2_perceptual_report_2021.md


⸻

4. Generating the Four Temporal Conditions (W4)

W4 investigates four temporal regimes.
The mapping stays fixed; the conditions under which the mapped durations unfold are varied.

4.1 Produce the modeled dt sequences

python scripts/w4_param_explorer.py \
    --events data/events_2021.csv \
    --baseline baseline/W2_baseline.json \
    --outdir reports/W4_variants/2021

This creates three modeled variants:
	•	REF/ — MODELED
	•	INERTIA/ — high ε
	•	FRICTION/ — high λ

The EMPIRICAL variant uses dt_applied and does not require regeneration.

⸻

5. Rendering the 60-Second Videos

Frame allocation converts each duration $\Delta t_i$ into a repetition count $n_i$.
Brightness is modulated by $\rho$.

EMPIRICAL

python scripts/w4_render_driver.py \
  --csv reports/W3_logs/2021/events_merged_2021__w3-baseline-run.csv \
  --dt-col dt_applied \
  --auto-seconds 60 --fps 30 \
  --mode brightness --text \
  --out out/W4_REF-empirical_2021.mp4

MODELED

python scripts/w4_render_driver.py \
  --csv reports/W4_variants/2021/REF/events_dt_model.csv \
  --dt-col dt_model \
  --auto-seconds 60 --fps 30 \
  --mode brightness --text \
  --out out/W4_REF-modeled_2021.mp4

INERTIA

python scripts/w4_render_driver.py \
  --csv reports/W4_variants/2021/INERTIA/events_dt_model.csv \
  --dt-col dt_model \
  --auto-seconds 60 --fps 30 \
  --mode brightness --text \
  --out out/W4_INERTIA_2021.mp4

FRICTION

python scripts/w4_render_driver.py \
  --csv reports/W4_variants/2021/FRICTION/events_dt_model.csv \
  --dt-col dt_model \
  --auto-seconds 60 --fps 30 \
  --mode brightness --text \
  --out out/W4_FRICTION_2021.mp4

All outputs appear in:

out/


⸻

6. Integrity and Verification

The artifact includes:
	•	environment.yml
	•	CHECKSUMS.txt
	•	all scripts for W2 & W4
	•	audited W2 reports
	•	regenerated W4 videos

Verify integrity:

sha256sum -c CHECKSUMS.txt


⸻

7. Scope of the Artifact

Reproduced here:
	•	calibrated temporal regime (ε*, λ*)
	•	three modeled temporal conditions
	•	the empirical condition
	•	all perceptual video renderings

Excluded (by design):
	•	embedding computation
	•	raw corpus
	•	multi-year datasets
	•	exploratory development history

These exclusions follow data-use constraints and are not necessary for auditability.

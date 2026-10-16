## Full Analysis

Run all project analyses (survey, wavelength game, and profile-vs-behavior correlation) for all four populations.

### Outputs

All outputs are written under:

- `Full Analysis/results/survey_report/`
  - `survey_<pop>.html` (4 HTML reports)
  - `survey_<pop>_csv/` (CSV exports)

- `Full Analysis/results/wgame_report/`
  - `wavelength_<pop>.html` (4 HTML reports)
  - `wavelength_<pop>_csv/` (CSV exports)

- `Full Analysis/results/correlation-results/<pop>/`
  - `phase1_correlations.csv`
  - `phase2_3_multivariate_results.csv`
  - (optional) `scatter_header_*.png` if matplotlib/seaborn are available

Populations (`<pop>`): `mistral-nemo-2407`, `llama3.1-8b`, `7b`, `14b`.

### Run

From repo root:

```bash
./"Full Analysis"/run_full_analysis.sh
```

This is CPU-only (no GPU required).

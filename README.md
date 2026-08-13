# WOWERS

**A national screening platform for micro-hydropower recovery at U.S. wastewater outfalls.**

Master's thesis, University of St. Thomas, August 2026.
Xinsheng Tang · Mohamed Abdel Hamid. Advisor: Dr. Kundan Nepal.

The full write-up is `thesis/WOWERS_Thesis.tex`.

---

## What this is

Every publicly owned treatment works (POTW) in the United States discharges treated
effluent continuously, at a flow already metered for permit compliance, and often over a
usable elevation drop. That hydraulic energy is discarded. No prior assessment had
quantified how much of it is recoverable plant by plant, because the cost of searching
seventeen thousand plants exceeded the value of the answer.

This repository screens the national fleet from public regulatory data alone. A
four-phase pipeline ingests ~279 million EPA discharge monitoring records, derives a flow
duration curve per plant, converts it to an annual-energy distribution by Monte-Carlo
sampling, estimates net head from the USGS 3D Elevation Program, matches a turbine from a
vendor catalogue, and builds a thirty-year financial scorecard. Results are published as a
deterministic GeoJSON contract that a static React/MapLibre dashboard renders as an
interactive national map.

**It is a screening instrument, not a validated predictor.** No site in the portfolio has
been built or metered. Every energy figure is modeled, and the headline is reported as a
calibrated band rather than a point estimate.

## Headline result

| | |
|---|---|
| POTWs screened | 17,148 |
| → flow-valid → head-valid → turbine-viable | 5,464 → 4,860 → 3,778 |
| **Project-viable sites** | **1,138** |
| Rated capacity | 58.59 MW |
| Physics ceiling | 409.17 GWh/yr |
| **Calibrated band** | **89.8–281 GWh/yr** |
| Portfolio NPV / CapEx / revenue | $310.13M / $211.33M / $41.23M/yr |
| Median payback | 9.83 yr |

The band matters more than the ceiling. The floor rests on Point Loma, the only metered
treated-wastewater conduit plant in the country (capacity factor 0.1914), corroborated
from above at 114.4 GWh/yr by 115 metered conduit plants and at 118.9 GWh/yr by the
river-hydropower first quartile. The upper tier rests on a single project's *design
projection*, not a measurement, and is labeled optimistic throughout. See §4.4.1 and
§5.1.4 of the thesis.

A supervised model was designed to learn a correction against measured generation and was
**abandoned against a pre-set gate**: a national label search returned 11 usable
conduit-scale sites against a requirement of 50, and no label carries head or flow. That
negative result is reported in §4.4.2 and §5.1.5 rather than omitted.

## Repository layout

```
src/phase1/     POTW filtering, DMR flow features, composite ranking
src/phase2/     Monte-Carlo energy estimation (site-keyed seeding)
src/phase3/     Outfall coordinates, 3DEP head estimation, turbine selection
src/phase4/     Cost models, financial scorecard, calibrated energy tiers
src/phase5/     Ground-truth ingest, feature matrix, leakage lock, CV harness
scripts/        Capacity-factor calibration; GeoJSON export
config/         settings.yaml (all thresholds and coefficients), energy intensity, state rates
exports/        The 59-property GeoJSON data contract (git-tracked, feeds the dashboard)
frontend/       React 19 + Vite 7 + MapLibre GL 5 dashboard, seven routed views
tests/          685 tests
thesis/         LaTeX source, 20 figures, and the script that regenerates each figure
```

## Running it

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # Python 3.11+
```

### Tests

```bash
python -m pytest tests/ -q
# 685 passed, 20 skipped
```

The 20 skips are integration tests guarded on `data/processed/*.parquet`, which is not
distributed (see below). They skip cleanly rather than fail.

### Pipeline

Each phase writes a parquet the next one reads, so any phase re-runs independently.

```bash
python -m src.phase1.run --skip-download   # requires the raw EPA data, see below
python -m src.phase2.run
python -m src.phase3.run
python -m src.phase4.run
python scripts/export_geojson.py           # writes both files in exports/
```

### Dashboard

No backend. The frontend reads the tracked GeoJSON directly.

```bash
cd frontend && npm install && npm run dev
```

### Thesis

```bash
cd thesis && tectonic -X compile WOWERS_Thesis.tex
```

Figures regenerate from the parquets:
`PYTHONPATH=. python3 thesis/figures/make_figNN_*.py`

## Data not included in this repository

`data/` is gitignored. The pipeline's inputs are public but large, and are downloaded
rather than committed:

| Source | Volume |
|---|---|
| EPA ECHO `npdes_downloads.zip` (ICIS facilities + permits) | ~322 MB |
| EPA ECHO DMR, FY2009–2024 | ~9.6 GB, ~279M rows |
| EPA ECHO `npdes_outfalls_layer.zip` (discharge coordinates) | ~50 MB |
| USGS 3DEP elevation | queried on demand, disk-cached |
| DOE HydroSource EHA + EIA-860/923 (calibration labels) | workbooks |

URLs are in `config/settings.yaml` under `epa.*` and `usgs.*`. Phase 1 downloads them on
first run; pass `--skip-download` if they are already on disk.

**Two paths in `config/settings.yaml` (`phase5.eia_data_dir`, `phase5.eha_data_dir`) and
one default in `scripts/cf_calibration.py` point at an external drive, `/Volumes/SANDISK/…`,
used during development.** They are defaults only. Override them in the config, or pass
`--eha-dir` to the calibration script. Nothing in Phases 1–4 depends on them.

The exported results *are* committed — `exports/scored_sites.geojson` (3,778 sites) and
`exports/viable_sites.geojson` (1,138 sites), 59 properties each — so the dashboard and
every figure work without re-running the pipeline. The full property dictionary is
Appendix A of the thesis.

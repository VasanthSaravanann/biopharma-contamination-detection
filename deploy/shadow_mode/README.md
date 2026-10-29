Shadow-mode deployment (scaffold)

This folder contains minimal scaffolding to run the pipeline in shadow mode (alerts logged but not actioned).

Usage
-----
1. Activate virtualenv: `source .venv/bin/activate`
2. Run the shadow script:

```bash
bash deploy/shadow_mode/run_shadow_mode.sh --config config/pipeline_config.yaml --out output/shadow
```

Behavior
--------
- The script runs the processing and scoring pipeline on live or replayed data and writes scores to `output/shadow/`.
- Alerts are recorded to `output/shadow/alerts.log` but not sent to external systems.

**Operational Runbook & Alert Rules**

Overview
--------
This runbook lists operational thresholds, alerting rules, and procedures for on-call response.

Key alert rules
---------------
- **Contamination alert:** model anomaly score > threshold for 2 consecutive samples → P1 alert to production team.
- **Drift amber:** `drift_acceptance_ratio` < 0.75 → P2 investigation ticket.
- **Drift red:** `drift_acceptance_ratio` < 0.40 → P1, suspend automated alerts, notify ML ops lead.

Example Prometheus-style alert (informational)
----------------------------------------------
```
ALERT ContaminationDetected
  IF avg_over_time(anomaly_score{job="rcde"}[5m]) > 0.7
  FOR 5m
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "Sustained high anomaly score",
    description = "Anomaly score > 0.7 for 5 minutes. Check sample and run on-call playbook."
  }
```

On-call playbook (high level)
------------------------------
1. Verify sample timestamps and sensor health in `output/shadow/alerts.log`.
2. Initiate confirmatory sampling (culture plating and qPCR) for the affected run.
3. If ground truth confirms contamination, follow batch abort / containment procedures and escalate to QA.
4. If false alarm, document and adjust thresholds or add context features.

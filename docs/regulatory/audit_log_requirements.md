**Audit Log Requirements (GMP-Ready Template)**

Minimum fields for audit logs:
- Timestamp (ISO8601)
- Actor (user/service id)
- Action (model_score, retrain, drift_detected, alert_ack)
- Input snapshot reference (path or object key)
- Model version (git commit or model hash)
- Outcome (score, decision)

Storage & retention
-------------------
- Immutable append-only storage recommended (S3 with Object Lock or equivalent).
- Retention policy: ≥5 years for regulated runs.

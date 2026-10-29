# Multi-site Manifest Template

Use this template to register site-level metadata for multi-site validation.

Columns:
- site_id: Short identifier for the site
- institution: Institution or company name
- instrument_model: Instrument or bioreactor model
- n_batches: Number of batches contributed
- organisms: Semicolon-separated organism list
- inocula: Semicolon-separated inoculum levels (CFU/mL)
- data_root: Path to site data root
- contact: Email of site lead

Example CSV rows:

site_id,institution,instrument_model,n_batches,organisms,inocula,data_root,contact
SITE_A,Institute A,AMBR-15,20,E_coli;B_subtilis,1;10;100,/data/site_a,lead@insta.edu
SITE_B,Industrial Partner,Custom_Bioreactor,25,E_coli;C_albicans,10;100,/data/site_b,ops@partner.com

Pre-registration checklist:
- Primary endpoint: ROC-AUC on held-out batches
- Secondary endpoints: detection limit at 10 CFU/mL, time-to-detection within 30 minutes
- Split strategy: batch-level holdout by site (train on sites A/B, test on C)
- Statistical plan: bootstrap 95% CI with 1000 iterations; random-effects meta-analysis for heterogeneity

REVIEWER REPRODUCIBILITY PACKAGE
================================
Satisfaction survey (Quality in Use) of an Academic Information System (AIS): anonymized data, analysis code, simulation code, seeds, and software versions.
Manuscript (IJIES, Paper ID 20267219): "Benchmark-First K-Means Check for Threshold-Only Clusters in Satisfaction Data of an Academic Information System"
(M. F. Rifai, P. Purwanto, D. M. K. Nugraheni). The package is publicly available in this GitHub repository; a Zenodo archive with a DOI will be added upon acceptance.

Quick start
-----------
pip install -r requirements.txt          # Python 3.10, versions in requirements.txt
Scripts in the main folder run from the main folder (python analysis_qiu_clustering.py, ...). Scripts in revision/ run from revision/ and take the data file as the first argument, e.g.
cd revision
python stability_uniform.py ../qiu_ais_survey_anonymized.csv 200

Where each manuscript result comes from
---------------------------------------
Data and item information
  qiu_ais_survey_anonymized.csv   314 respondents, 22 items, background variables, reported problems (UTF-8).
  data_dictionary.csv             variables, module, characteristic each item was written for, Indonesian item text, English translation (manuscript Table A1).
Survey analyses (Sections 3.1 to 3.10)
  analysis_qiu_clustering.py (+ results_summary.txt), efa_parallel_map.py, efa_loadings_scree.py (Table 2), gmm_bic_check.py, overall_definition_*.py, scaling_*.py, confidence_intervals.py, logs/.
  revision/ordinal_cfa.py            polychoric ULS refit, latent correlations, SRMR (Section 3.1)   -> revision/results/ordinal_cfa_results.txt, ordinal_loadings_*.csv, polychoric_matrix.csv
  revision/k2_vs_k3.py               two versus three groups, cross-tabulations (Section 3.2)        -> revision/results/k2_vs_k3_results.txt
  revision/stability_uniform.py, stability_kmeans_k.py   uniform bootstrap stability (Table 3, Section 3.2) -> revision/results/stability_*.txt
  revision/survey_residual.py, survey_residual_test.py   residual variation, partition cross-tabulations, monotonicity, conditional residual-profile test (Section 3.3) -> revision/results/survey_residual*.txt
Baseline simulation (Section 3.4, Tables 4 and 5)
  simulate_benchmark_first.py -> simulation_results.csv (Table 4);  compare_established_methods.py -> method_comparison_results.csv (Table 5, columns ARI rule to Silh.)
  revision/residual_comparator_table5.py -> revision/results/residual_comparator_table5_results.txt (Table 5, column Resid.)
Extended simulation, calibration and validation of the ARI threshold (Sections 2.11 and 3.11, Tables 7 and 8)
  revision/sim_extended.py           240 configurations x 200 replications x k = 2, 3, 4  -> revision/results/sim_extended_raw.csv (144,000 rows), sim_extended_configs.csv
  revision/calibrate_tau.py          calibration half / validation half, loss, intervals  -> revision/results/calibration_full_output.txt, tau_grid_results.csv
  Example: python sim_extended.py --reps 200 --configs 240 --start 0 --end 60 --out chunk0.csv --jobs 4   (run the four chunks 0-60, 60-120, 120-180, 180-240 and concatenate to reproduce sim_extended_raw.csv; about 5 to 10 minutes per chunk on 4 cores)
Figures
  revision/make_figures.py -> revision/results/figures (Figs. 1 to 5)
Deduplication (Section 3.8)
  deduplication_sensitivity_summary.csv and deduplication_facts.csv give the sensitivity results and counts WITHOUT names or identifiers. The raw 401-row file contains names and is not shared, so the deduplication itself cannot be rerun; the shared file holds the 314 retained responses.
Definitions and settings
  BOOTSTRAP_DEFINITIONS.txt   resampling, scaling, assignment and comparison set, paired tertile difference
  SEEDS_AND_SETTINGS.txt      all seeds, clustering settings, comparator tuning, software versions

Notes
-----
- Overall score = mean of the four characteristic composites (EF, EC, ST, CC), each the mean of its items. K-Means: k-means++, random_state = 42 (survey), min-max normalized composites. The ARI rule compares K-Means on the four composites with K-Means on the overall score alone; ARI >= tau = 0.80 is read as thresholds (an operational reporting convention, not a validated test of dimensional structure).
- Row order matters for exact reproduction of k-means results; use the file as provided.
- Bootstrap stability in analysis_qiu_clustering.py (k = 2, 3, 4: 0.944, 0.918, 0.786) uses the older protocol (scaling not re-estimated). The revised manuscript reports the uniform protocol (0.945, 0.924, 0.781; revision/stability_*.py).
- Cramer's V for study program and semester uses the grouped categories in the shared file; the manuscript reports these grouped values.
- Table 2 (exploratory factor analysis) comes from efa_loadings_scree.py (principal axis, oblimin). efa_parallel_map.py is an exploratory companion with a different extraction, so its loadings and factor correlation (0.73) differ slightly from Table 2.
- Free-text answers and the raw 401-row file (names) are not shared for privacy.

License
-------
Data and code: Creative Commons Attribution 4.0 International (CC BY 4.0). See LICENSE.txt.

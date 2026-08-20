# Planned DirectProbe pause

- Pause requested: 2026-08-19T20:45:46+02:00
- Original runner start: approximately 2026-08-19T16:30:41+02:00
- Environment: `attention`
- Model: `codebert`
- Dataset size: 3,000 programs per language
- Languages: Java, Go, JavaScript
- Requested layers: 5, 9, 12
- Workers per active configuration: 10
- Per-configuration timeout: 43,200 seconds
- Required solver: Gurobi
- Stop method: SIGTERM to the six runners and active `main.py` children, followed by SIGTERM to the exact orphaned Joblib worker PIDs
- Post-stop verification: no matching runner, `main.py`, Joblib worker, or resource-tracker process remained

## Completed results retained in the normal results tree

- Java `siblings`, layer 5
- Go `siblings`, layers 5 and 9
- JavaScript `siblings`, layer 5

Each retained directory contains `clusters.txt`, `prediction.txt`, `dis.txt`, and `log.txt`.

## Incomplete results archived here

- Java `siblings`, layer 9
- Java `distance`, layer 5
- Go `siblings`, layer 12
- Go `distance`, layer 5
- JavaScript `siblings`, layer 9
- JavaScript `distance`, layer 5

The archived directories contain partial logs only. They must not be treated as valid DirectProbe results and must be rerun from the beginning.

## Runner queues that were stopped

For each language, one runner covered `siblings siblings_id dfg` and another covered `distance distance_id`, at layers 5, 9, and 12. All queued configurations not listed as completed above were not started or not completed.

The Gurobi runs used the explicitly supplied project-local `GRB_LICENSE_FILE`; no global Gurobi configuration was changed.

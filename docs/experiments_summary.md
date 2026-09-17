# Experiment Summary

The development archive retains detailed reports and all historical runs locally.
The public mirror publishes only this compact summary.

| Stage | Decision | Status |
| --- | --- | --- |
| Annotation curation | Parent-space review, recovery, and grid-tile projection | Incorporated into audit-fixed v2 |
| Coverage audit | Deterministic duplicate and missing-coverage repair with explicit ledgers | Incorporated into audit-fixed v2 |
| Filtering | v9 aggressive small/low/xView filtering while retaining `bad` parents | Active policy |
| Resolution | 1280 selected over the tested lower resolutions | Active policy |
| Augmentation | `degrees=180`, `flipud=0.5`, `fliplr=0.5` carried forward | Selected recipe |
| HPO | Historical and bounded audit-fixed trials retained as evidence | No automatic claim beyond validation split |

The selected release metadata reports mAP50-95 `0.70666`, mAP50 `0.89688`, precision
`0.95672`, and recall `0.86197` on the project validation split. These values are not
an independent test-set result and should not be compared across incompatible data
splits or label policies.

The release process retains pre-audit experiments as historical evidence only. Model
selection should be revalidated if the active dataset, split, annotation policy,
hardware, or Ultralytics version changes.

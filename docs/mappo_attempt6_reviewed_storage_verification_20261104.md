# Storage verification after complete native replay

The registered collectors, independent controller and physics reviewers,
matching setup, inference functions, sample schedule and criteria are unchanged.
The postprocessing completion auditor now avoids decoding every trace again
when it only needs to establish identity with already reviewed data.

Before using this storage path, the completion auditor still requires the
actual successful root wait, every exact collector/reviewer native exit, the
review-verification hash and the collector-manifest hash. Each native reviewer
has already decoded and checked every payload and independently executed every
controller decision and physical step. The storage check verifies the whole
database and seal SHA256, original block ordering, all scheduled case/method
keys, SQLite integrity and complete sealed row/trace checksum catalog. Under
the same SHA256 integrity assumption used throughout the project, unchanged
database bytes preserve that earlier complete replay evidence.

The six storage regression cases cover a valid exact archive without redundant
decoding, a database bit flip, missing/duplicate/foreign schedule entries, and
a forged trace-hash catalog. All passed locally. On all 43008 calibration
records whose eight native collection/review exits had completed successfully,
the storage verifier checked every block in 7.329 seconds. This is a storage
operation measurement, not an online controller speedup experiment.

The paper exporter still reads every completed trajectory to reconstruct exact
zero-adoption fallback, pre-adoption prefixes and actual role participation.
The original full native replay remains mandatory for all 59392 planned
method-initial records. No partial primary effects were inspected for this
change. No new statistical inference, sample omission, controller change,
threshold change or error allocation follows from this optimization.

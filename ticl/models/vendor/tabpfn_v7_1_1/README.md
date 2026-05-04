TabPFN Regressor Vendor Snapshot
================================

Upstream source of truth:
- Repository: `PriorLabs/TabPFN`
- Release tag: `v7.1.1`
- Release date: `2026-04-09`
- Commit: `0be2a61671ce8515d8f5c10f44bf7911aa15c395`

Rules for this vendor directory:
- `upstream/src/tabpfn/...` contains exact source snapshots copied from the official
  release.
- Runtime glue in this directory must stay thin and must point back to those
  snapshots as the source of truth.
- If upstream behavior changes, update the snapshot first, then update the thin
  runtime glue and the numeric parity tests.

Current exact upstream snapshots:
- `upstream/src/tabpfn/architectures/base/bar_distribution.py`
- `upstream/src/tabpfn/finetuning/finetuned_regressor.py`
- `upstream/src/tabpfn/utils.py`
- `upstream/src/tabpfn/regressor.py`
- `upstream/src/tabpfn/architectures/tabpfn_v2_6.py`
- `upstream/src/tabpfn/architectures/interface.py`
- `upstream/src/tabpfn/constants.py`

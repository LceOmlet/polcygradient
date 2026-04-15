# Tracked External Artifacts

This directory stores a Git-tracked snapshot of external artifacts that were
originally written under `/home/chen/RLPFN/artifacts`.

Rules for this snapshot:

- Files with size `<= 3 MiB` are mirrored into
  `tracked_artifacts/external_artifacts/`.
- Files with size `> 3 MiB` are not committed, but they are still recorded in
  `tracked_artifacts/external_artifacts_manifest.json` with original path,
  size, and `sha256`.
- The manifest is the traceability source of truth for both copied and
  excluded files.

Refresh the snapshot with:

```bash
python ticl/analysis/snapshot_external_artifacts.py \
  --source-root /home/chen/RLPFN/artifacts \
  --repo-root /home/chen/RLPFN/reinforce-terminal-explore
```

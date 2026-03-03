# Rebuilt Skyline Ladder (post-stability)

## Validity Gate

- `strict_valid = True` iff all conditions hold:
  - `status == ok` in `[pg-phase]`
  - `valid/skipped` is `>=1/0`
  - `mean loss` is finite
- Purpose: prevent pre-stability `grad_norm_nonfinite/inf` runs from polluting skyline comparisons.

## Summary

- Runs scanned: `98`
- Strict-valid runs: `4`
- Non-strict/diagnostic runs: `94`
- Status counts: `{"grad_norm_nonfinite": 77, "ok": 4, "unknown": 16, "oom": 1}`

## Strict-Valid Skyline

| rank | run_id | status | valid/skipped | wallclock(s) | rollout/backward(s) | chunk | wall/chunk | dtype | notes |
|---:|---|---|---|---:|---|---:|---:|---|---|
| 1 | `20260303_012000_denseprefixcache_default_probe_rep2` | `ok` | `1/0` | `48.36` | `48.29/27.79` | `8` | `6.04` | `fp16` | - |
| 2 | `20260303_093756_post_rebuild_transition_profile_ok` | `ok` | `1/0` | `49.47` | `49.41/27.88` | `8` | `6.18` | `bf16` | fin=48.4%, tr(y/x)=33.7/33.4% |
| 3 | `20260302_132022_syncnanopt` | `ok` | `1/0` | `73.75` | `73.66/43.94` | `8` | `9.22` | `na` | - |
| 4 | `20260302_235300_denseprefix_page48_fixedcfg_probe` | `ok` | `1/0` | `96.90` | `96.82/60.97` | `8` | `12.11` | `fp16` | - |

## Diagnostic Ladder (Invalid/Skipped/OOM)

| rank | run_id | status | valid/skipped | wallclock(s) | rollout/backward(s) | chunk | wall/chunk | dtype | notes |
|---:|---|---|---|---:|---|---:|---:|---|---|
| 1 | `20260303_033200_tf_layer_deep_profile_after_finalize2d` | `oom` | `0/1` | `14.75` | `0.00/0.00` | `32` | `0.46` | `bf16` | fin=56.1% |
| 2 | `20260303_004300_default_after_densecap48_rep2` | `grad_norm_nonfinite` | `0/1` | `49.73` | `49.68/28.33` | `8` | `6.22` | `fp16` | - |
| 3 | `20260303_000800_dense_pagesize48_fixedcfg_probe` | `grad_norm_nonfinite` | `0/1` | `50.07` | `50.02/28.21` | `8` | `6.26` | `fp16` | - |
| 4 | `20260302_230500_qkvtruefusion_fixedcfg` | `grad_norm_nonfinite` | `0/1` | `51.12` | `51.08/29.02` | `8` | `6.39` | `fp16` | tr(y/x)=34.1/33.6% |
| 5 | `20260303_025200_finalize2d_fastpath_seeded` | `grad_norm_nonfinite` | `0/1` | `52.70` | `52.66/29.98` | `8` | `6.59` | `fp16` | - |
| 6 | `20260303_030500_finalize2d_ab_on` | `grad_norm_nonfinite` | `0/1` | `52.90` | `52.85/29.74` | `8` | `6.61` | `fp16` | - |
| 7 | `20260303_010400_seeded_densecap128_ab` | `grad_norm_nonfinite` | `0/1` | `53.45` | `53.41/30.01` | `8` | `6.68` | `fp16` | - |
| 8 | `20260303_023200_pre_sdpa_backend_seeded` | `grad_norm_nonfinite` | `0/1` | `53.67` | `53.62/30.03` | `8` | `6.71` | `fp16` | - |
| 9 | `20260303_032500_finalize2d_noseed_off_control` | `grad_norm_nonfinite` | `0/1` | `53.86` | `53.81/30.33` | `8` | `6.73` | `fp16` | - |
| 10 | `20260303_010000_seeded_densecap48_ab` | `grad_norm_nonfinite` | `0/1` | `54.16` | `54.11/30.80` | `8` | `6.77` | `fp16` | - |
| 11 | `20260303_005200_default_densecap128_ab` | `grad_norm_nonfinite` | `0/1` | `54.20` | `54.15/30.46` | `8` | `6.78` | `fp16` | - |
| 12 | `20260303_004700_default_after_densecap48_rep3` | `grad_norm_nonfinite` | `0/1` | `54.29` | `54.25/30.91` | `8` | `6.79` | `fp16` | - |
| 13 | `20260303_030900_finalize2d_ab_off` | `grad_norm_nonfinite` | `0/1` | `54.32` | `54.27/30.57` | `8` | `6.79` | `fp16` | - |
| 14 | `20260303_025700_finalize2d_fastpath_seeded_rep2` | `grad_norm_nonfinite` | `0/1` | `54.35` | `54.30/30.59` | `8` | `6.79` | `fp16` | - |
| 15 | `20260303_031700_finalize2d_noseed_maincmd` | `grad_norm_nonfinite` | `0/1` | `54.69` | `54.64/31.04` | `8` | `6.84` | `fp16` | - |
| 16 | `20260303_010800_seeded_densecap128_bf16_probe` | `grad_norm_nonfinite` | `0/1` | `55.29` | `55.24/31.44` | `8` | `6.91` | `bf16` | - |
| 17 | `20260303_001500_dense_pagesize48_fixedcfg_probe_rep3` | `grad_norm_nonfinite` | `0/1` | `55.30` | `55.26/31.46` | `8` | `6.91` | `fp16` | - |
| 18 | `20260303_004000_default_after_densecap48` | `grad_norm_nonfinite` | `0/1` | `55.67` | `55.62/32.27` | `8` | `6.96` | `fp16` | - |
| 19 | `20260302_231000_qkvtruefusion_fixedcfg_plain` | `grad_norm_nonfinite` | `0/1` | `55.77` | `55.72/31.59` | `8` | `6.97` | `fp16` | - |
| 20 | `20260302_234000_transition_streamfusion_probe` | `grad_norm_nonfinite` | `0/1` | `56.51` | `56.47/32.06` | `8` | `7.06` | `fp16` | - |
| 21 | `20260303_024200_tf_layer_deep_profile` | `grad_norm_nonfinite` | `0/1` | `56.93` | `56.88/32.52` | `8` | `7.12` | `fp16` | fin=52.4%, tr(y/x)=38.5/37.8% |
| 22 | `20260303_002300_plain_after_layerprofile_code` | `grad_norm_nonfinite` | `0/1` | `56.99` | `56.94/32.22` | `8` | `7.12` | `fp16` | - |
| 23 | `20260302_231300_qkvtruefusion_fixedcfg_plain_rep2` | `grad_norm_nonfinite` | `0/1` | `57.14` | `57.09/32.66` | `8` | `7.14` | `fp16` | - |
| 24 | `20260303_001200_dense_pagesize48_fixedcfg_probe_rep2` | `grad_norm_nonfinite` | `0/1` | `57.52` | `57.47/32.75` | `8` | `7.19` | `fp16` | - |
| 25 | `20260303_002700_dense_pagesize48_fixedcfg_probe_rep4` | `grad_norm_nonfinite` | `0/1` | `57.67` | `57.62/33.37` | `8` | `7.21` | `fp16` | - |
| 26 | `20260303_000100_fixedcfg_tf_layer_profile` | `grad_norm_nonfinite` | `0/1` | `58.07` | `58.03/33.09` | `8` | `7.26` | `fp16` | tr(y/x)=38.1/38.0% |
| 27 | `20260303_003500_dense_pagesize32_fixedcfg_probe` | `grad_norm_nonfinite` | `0/1` | `58.62` | `58.57/33.20` | `8` | `7.33` | `fp16` | - |
| 28 | `20260303_001900_pagesize48_tf_layer_profile` | `grad_norm_nonfinite` | `0/1` | `58.79` | `58.74/33.62` | `8` | `7.35` | `fp16` | tr(y/x)=39.2/39.6% |
| 29 | `20260303_003100_dense_pagesize64_fixedcfg_probe` | `grad_norm_nonfinite` | `0/1` | `59.34` | `59.30/33.08` | `8` | `7.42` | `fp16` | - |
| 30 | `20260303_011600_denseprefixcache_default_probe` | `grad_norm_nonfinite` | `0/1` | `60.46` | `60.41/34.85` | `8` | `7.56` | `fp16` | - |

## Cohort Ladder

| rank | cohort_key | runs | strict_valid | best_strict_valid(s) | best_any_status(s) | best_valid_run | best_any_run |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `6` | `1` | `48.36` | `48.36` | `20260303_012000_denseprefixcache_default_probe_rep2` | `20260303_012000_denseprefixcache_default_probe_rep2` |
| 2 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=True` | `1` | `1` | `49.47` | `49.47` | `20260303_093756_post_rebuild_transition_profile_ok` | `20260303_093756_post_rebuild_transition_profile_ok` |
| 3 | `seeded=0|dtype=na|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `5` | `1` | `73.75` | `73.75` | `20260302_132022_syncnanopt` | `20260302_132022_syncnanopt` |
| 4 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `23` | `1` | `96.90` | `51.12` | `20260302_235300_denseprefix_page48_fixedcfg_probe` | `20260302_230500_qkvtruefusion_fixedcfg` |
| 5 | `seeded=1|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=32|tbptt=1|fin2d=True` | `1` | `0` | `na` | `14.75` | `na` | `20260303_033200_tf_layer_deep_profile_after_finalize2d` |
| 6 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=48|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `3` | `0` | `na` | `49.73` | `na` | `20260303_004300_default_after_densecap48_rep2` |
| 7 | `seeded=0|dtype=fp16|kv=paged|page=48|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `5` | `0` | `na` | `50.07` | `na` | `20260303_000800_dense_pagesize48_fixedcfg_probe` |
| 8 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `8` | `0` | `na` | `52.70` | `na` | `20260303_025200_finalize2d_fastpath_seeded` |
| 9 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=48|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `1` | `0` | `na` | `54.16` | `na` | `20260303_010000_seeded_densecap48_ab` |
| 10 | `seeded=1|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `1` | `0` | `na` | `55.29` | `na` | `20260303_010800_seeded_densecap128_bf16_probe` |
| 11 | `seeded=0|dtype=fp16|kv=paged|page=32|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `1` | `0` | `na` | `58.62` | `na` | `20260303_003500_dense_pagesize32_fixedcfg_probe` |
| 12 | `seeded=0|dtype=fp16|kv=paged|page=64|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `1` | `0` | `na` | `59.34` | `na` | `20260303_003100_dense_pagesize64_fixedcfg_probe` |
| 13 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `5` | `0` | `na` | `66.92` | `na` | `20260302_154600_rolloutdtype_auto_seeded` |
| 14 | `seeded=1|dtype=na|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na` | `5` | `0` | `na` | `71.07` | `na` | `20260302_133327_baseline_seeded_wt189d9f3` |
| 15 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=32|tbptt=128|fin2d=na` | `1` | `0` | `na` | `76.23` | `na` | `20260302_165800_flashprefix_bs32` |
| 16 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=68|tbptt=128|fin2d=na` | `7` | `0` | `na` | `79.94` | `na` | `20260302_181900_flashprefix_bs68_page48` |
| 17 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=64|tbptt=128|fin2d=na` | `3` | `0` | `na` | `82.34` | `na` | `20260302_180500_flashprefix_bs64_page48` |
| 18 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=70|tbptt=128|fin2d=na` | `4` | `0` | `na` | `84.69` | `na` | `20260302_182600_flashprefix_bs70_page48_oomprobe` |
| 19 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=48|tbptt=128|fin2d=na` | `1` | `0` | `na` | `87.12` | `na` | `20260302_171500_flashprefix_bs48` |
| 20 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=na|tbptt=na|fin2d=na` | `11` | `0` | `na` | `102.75` | `na` | `20260302_160200_tbpttbwd_auto` |
| 21 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=na|tbptt=na|fin2d=na` | `1` | `0` | `na` | `na` | `na` | `na` |
| 22 | `seeded=0|dtype=na|kv=na|page=na|densecap=na|ckpt=na|reentrant=na|mutable=na|compile=na|chunk=na|tbptt=na|fin2d=na` | `1` | `0` | `na` | `na` | `na` | `na` |
| 23 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=na|tbptt=na|fin2d=na` | `1` | `0` | `na` | `na` | `na` | `na` |
| 24 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=na|tbptt=na|fin2d=True` | `2` | `0` | `na` | `na` | `na` | `na` |

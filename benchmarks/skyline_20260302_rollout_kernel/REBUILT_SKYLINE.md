# Rebuilt Skyline Ladder (post-stability)

## Validity Gate

- `strict_valid = True` iff all conditions hold:
  - `status == ok` in `[pg-phase]`
  - `valid/skipped` is `>=1/0`
  - `mean loss` is finite
- Skyline cohorts are also separated by `pg_loss_sig` to avoid mixing different loss forms.
- Purpose: prevent pre-stability `grad_norm_nonfinite/inf` runs from polluting skyline comparisons.

## Summary

- Runs scanned: `128`
- Strict-valid runs: `28`
- Non-strict/diagnostic runs: `100`
- Status counts: `{"grad_norm_nonfinite": 77, "ok": 28, "unknown": 21, "oom": 2}`

## Strict-Valid Skyline

| rank | run_id | status | valid/skipped | wallclock(s) | wall/batch(s) | rollout/backward(s) | chunk | wall/chunk | dtype | notes |
|---:|---|---|---|---:|---:|---|---:|---:|---|---|
| 1 | `20260303_012000_denseprefixcache_default_probe_rep2` | `ok` | `1/0` | `48.36` | `na` | `48.29/27.79` | `8` | `6.04` | `fp16` | - |
| 2 | `20260303_093756_post_rebuild_transition_profile_ok` | `ok` | `1/0` | `49.47` | `6.1837` | `49.41/27.88` | `8` | `6.18` | `bf16` | fin=48.4%, tr(y/x)=33.7/33.4% |
| 3 | `20260303_122704_bs8_auto_threshold_probe2` | `ok` | `1/0` | `53.87` | `6.7337` | `53.82/30.44` | `8` | `6.73` | `bf16` | fin=48.3%, tr(y/x)=37.7/37.2%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 4 | `20260303_113211_losssig_probe` | `ok` | `1/0` | `55.56` | `6.9450` | `55.51/31.68` | `8` | `6.95` | `bf16` | fin=47.6%, tr(y/x)=37.8/37.5%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 5 | `20260303_120432_bs64_auto_flashprefix_probe` | `ok` | `1/0` | `61.19` | `0.9561` | `61.15/34.16` | `64` | `0.96` | `bf16` | fin=7.7%, tr(y/x)=40.9/41.7%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 6 | `20260303_123618_bs8_fusedproj_inplace_probe` | `ok` | `1/0` | `62.63` | `7.8288` | `62.58/36.28` | `8` | `7.83` | `bf16` | fin=49.0%, tr(y/x)=41.6/40.6%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 7 | `20260303_122829_bs64_auto_threshold_probe3` | `ok` | `1/0` | `63.08` | `0.9856` | `63.03/35.75` | `64` | `0.99` | `bf16` | fin=7.4%, tr(y/x)=41.6/42.3%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 8 | `20260303_133459_bs64_qkvfuse_probe` | `ok` | `1/0` | `63.91` | `0.9986` | `63.86/36.48` | `64` | `1.00` | `bf16` | fin=46.6%, tr(y/x)=40.0/40.3%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 9 | `20260303_120616_bs8_auto_flashprefix_probe` | `ok` | `1/0` | `68.36` | `8.5450` | `68.31/39.72` | `8` | `8.54` | `bf16` | fin=6.4%, tr(y/x)=43.6/43.0%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 10 | `20260303_134234_bs64_cachecow_catkernel_probe` | `ok` | `1/0` | `68.59` | `1.0717` | `68.54/38.76` | `64` | `1.07` | `bf16` | fin=52.5%, tr(y/x)=43.0/44.1%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 11 | `20260303_122534_bs64_auto_threshold_probe2` | `ok` | `1/0` | `70.26` | `1.0978` | `70.21/39.72` | `64` | `1.10` | `bf16` | fin=7.5%, tr(y/x)=43.2/44.1%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 12 | `20260303_133916_bs64_qkvheads_layoutreuse_probe` | `ok` | `1/0` | `71.56` | `1.1181` | `71.51/41.62` | `64` | `1.12` | `bf16` | fin=45.6%, tr(y/x)=42.1/42.0%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 13 | `20260303_131129_bs64_cachecatfree_recover_probe2` | `ok` | `1/0` | `73.67` | `1.1511` | `73.62/43.14` | `64` | `1.15` | `bf16` | fin=37.9%, tr(y/x)=41.1/41.6%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 14 | `20260302_132022_syncnanopt` | `ok` | `1/0` | `73.75` | `na` | `73.66/43.94` | `8` | `9.22` | `na` | - |
| 15 | `20260303_124713_bs64_restore_after_inplace_revert` | `ok` | `1/0` | `74.42` | `1.1628` | `74.37/43.48` | `64` | `1.16` | `bf16` | fin=42.1%, tr(y/x)=40.6/41.8%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 16 | `20260303_132045_bs64_dispatch_profile_only_probe` | `ok` | `1/0` | `74.97` | `1.1714` | `74.92/43.80` | `64` | `1.17` | `bf16` | fin=37.9%, tr(y/x)=41.3/41.4%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 17 | `20260303_133218_bs64_finalize_revert_sanity_probe` | `ok` | `1/0` | `78.19` | `1.2217` | `78.15/45.97` | `64` | `1.22` | `bf16` | fin=37.8%, tr(y/x)=40.9/41.5%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 18 | `20260303_125514_bs64_cachecatfree_probe` | `ok` | `1/0` | `79.55` | `1.2430` | `79.50/46.86` | `64` | `1.24` | `bf16` | fin=40.9%, tr(y/x)=42.4/43.0%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 19 | `20260303_130000_bs64_projbhld_cachecatfree_probe` | `ok` | `1/0` | `82.72` | `1.2925` | `82.67/47.98` | `64` | `1.29` | `bf16` | fin=36.3%, tr(y/x)=42.3/44.0%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 20 | `20260303_132933_bs64_finalize_addmmfuse_probe` | `ok` | `1/0` | `87.29` | `1.3639` | `87.24/50.32` | `64` | `1.36` | `bf16` | fin=43.7%, tr(y/x)=42.4/43.4%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |

## Diagnostic Ladder (Invalid/Skipped/OOM)

| rank | run_id | status | valid/skipped | wallclock(s) | wall/batch(s) | rollout/backward(s) | chunk | wall/chunk | dtype | notes |
|---:|---|---|---|---:|---:|---|---:|---:|---|---|
| 1 | `20260303_131051_bs64_cachecatfree_recover_probe` | `oom` | `0/1` | `12.95` | `0.2023` | `0.00/0.00` | `32` | `0.40` | `bf16` | fin=37.2%, pg=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10 |
| 2 | `20260303_033200_tf_layer_deep_profile_after_finalize2d` | `oom` | `0/1` | `14.75` | `na` | `0.00/0.00` | `32` | `0.46` | `bf16` | fin=56.1% |
| 3 | `20260303_004300_default_after_densecap48_rep2` | `grad_norm_nonfinite` | `0/1` | `49.73` | `na` | `49.68/28.33` | `8` | `6.22` | `fp16` | - |
| 4 | `20260303_000800_dense_pagesize48_fixedcfg_probe` | `grad_norm_nonfinite` | `0/1` | `50.07` | `na` | `50.02/28.21` | `8` | `6.26` | `fp16` | - |
| 5 | `20260302_230500_qkvtruefusion_fixedcfg` | `grad_norm_nonfinite` | `0/1` | `51.12` | `na` | `51.08/29.02` | `8` | `6.39` | `fp16` | tr(y/x)=34.1/33.6% |
| 6 | `20260303_025200_finalize2d_fastpath_seeded` | `grad_norm_nonfinite` | `0/1` | `52.70` | `na` | `52.66/29.98` | `8` | `6.59` | `fp16` | - |
| 7 | `20260303_030500_finalize2d_ab_on` | `grad_norm_nonfinite` | `0/1` | `52.90` | `na` | `52.85/29.74` | `8` | `6.61` | `fp16` | - |
| 8 | `20260303_010400_seeded_densecap128_ab` | `grad_norm_nonfinite` | `0/1` | `53.45` | `na` | `53.41/30.01` | `8` | `6.68` | `fp16` | - |
| 9 | `20260303_023200_pre_sdpa_backend_seeded` | `grad_norm_nonfinite` | `0/1` | `53.67` | `na` | `53.62/30.03` | `8` | `6.71` | `fp16` | - |
| 10 | `20260303_032500_finalize2d_noseed_off_control` | `grad_norm_nonfinite` | `0/1` | `53.86` | `na` | `53.81/30.33` | `8` | `6.73` | `fp16` | - |
| 11 | `20260303_010000_seeded_densecap48_ab` | `grad_norm_nonfinite` | `0/1` | `54.16` | `na` | `54.11/30.80` | `8` | `6.77` | `fp16` | - |
| 12 | `20260303_005200_default_densecap128_ab` | `grad_norm_nonfinite` | `0/1` | `54.20` | `na` | `54.15/30.46` | `8` | `6.78` | `fp16` | - |
| 13 | `20260303_004700_default_after_densecap48_rep3` | `grad_norm_nonfinite` | `0/1` | `54.29` | `na` | `54.25/30.91` | `8` | `6.79` | `fp16` | - |
| 14 | `20260303_030900_finalize2d_ab_off` | `grad_norm_nonfinite` | `0/1` | `54.32` | `na` | `54.27/30.57` | `8` | `6.79` | `fp16` | - |
| 15 | `20260303_025700_finalize2d_fastpath_seeded_rep2` | `grad_norm_nonfinite` | `0/1` | `54.35` | `na` | `54.30/30.59` | `8` | `6.79` | `fp16` | - |
| 16 | `20260303_031700_finalize2d_noseed_maincmd` | `grad_norm_nonfinite` | `0/1` | `54.69` | `na` | `54.64/31.04` | `8` | `6.84` | `fp16` | - |
| 17 | `20260303_010800_seeded_densecap128_bf16_probe` | `grad_norm_nonfinite` | `0/1` | `55.29` | `na` | `55.24/31.44` | `8` | `6.91` | `bf16` | - |
| 18 | `20260303_001500_dense_pagesize48_fixedcfg_probe_rep3` | `grad_norm_nonfinite` | `0/1` | `55.30` | `na` | `55.26/31.46` | `8` | `6.91` | `fp16` | - |
| 19 | `20260303_004000_default_after_densecap48` | `grad_norm_nonfinite` | `0/1` | `55.67` | `na` | `55.62/32.27` | `8` | `6.96` | `fp16` | - |
| 20 | `20260302_231000_qkvtruefusion_fixedcfg_plain` | `grad_norm_nonfinite` | `0/1` | `55.77` | `na` | `55.72/31.59` | `8` | `6.97` | `fp16` | - |

## Cohort Ladder

| rank | cohort_key | runs | strict_valid | best_strict_valid(s) | best_any_status(s) | best_valid_run | best_any_run |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=64|tbptt=128|fin2d=True|pgsig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10` | `15` | `15` | `61.19` | `61.19` | `20260303_120432_bs64_auto_flashprefix_probe` | `20260303_120432_bs64_auto_flashprefix_probe` |
| 2 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=True|pgsig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10` | `4` | `4` | `53.87` | `53.87` | `20260303_122704_bs8_auto_threshold_probe2` | `20260303_122704_bs8_auto_threshold_probe2` |
| 3 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=64|tbptt=16|fin2d=True|pgsig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10` | `4` | `4` | `144.05` | `144.05` | `20260303_123336_bs64_fusedproj_inplace_probe` | `20260303_123336_bs64_fusedproj_inplace_probe` |
| 4 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `6` | `1` | `48.36` | `48.36` | `20260303_012000_denseprefixcache_default_probe_rep2` | `20260303_012000_denseprefixcache_default_probe_rep2` |
| 5 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=True|pgsig=na` | `1` | `1` | `49.47` | `49.47` | `20260303_093756_post_rebuild_transition_profile_ok` | `20260303_093756_post_rebuild_transition_profile_ok` |
| 6 | `seeded=0|dtype=na|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `5` | `1` | `73.75` | `73.75` | `20260302_132022_syncnanopt` | `20260302_132022_syncnanopt` |
| 7 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=64|tbptt=32|fin2d=True|pgsig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10` | `1` | `1` | `96.48` | `96.48` | `20260303_124926_bs64_auto_prefixfix_probe` | `20260303_124926_bs64_auto_prefixfix_probe` |
| 8 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `23` | `1` | `96.90` | `51.12` | `20260302_235300_denseprefix_page48_fixedcfg_probe` | `20260302_230500_qkvtruefusion_fixedcfg` |
| 9 | `seeded=0|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=32|tbptt=1|fin2d=True|pgsig=pg_v2|norm=0|disc=1|detach=1|eps=1e-06|clip=10|rclip=10` | `1` | `0` | `na` | `12.95` | `na` | `20260303_131051_bs64_cachecatfree_recover_probe` |
| 10 | `seeded=1|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=32|tbptt=1|fin2d=True|pgsig=na` | `1` | `0` | `na` | `14.75` | `na` | `20260303_033200_tf_layer_deep_profile_after_finalize2d` |
| 11 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=48|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `3` | `0` | `na` | `49.73` | `na` | `20260303_004300_default_after_densecap48_rep2` |
| 12 | `seeded=0|dtype=fp16|kv=paged|page=48|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `5` | `0` | `na` | `50.07` | `na` | `20260303_000800_dense_pagesize48_fixedcfg_probe` |
| 13 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `8` | `0` | `na` | `52.70` | `na` | `20260303_025200_finalize2d_fastpath_seeded` |
| 14 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=48|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `1` | `0` | `na` | `54.16` | `na` | `20260303_010000_seeded_densecap48_ab` |
| 15 | `seeded=1|dtype=bf16|kv=paged|page=128|densecap=128|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `1` | `0` | `na` | `55.29` | `na` | `20260303_010800_seeded_densecap128_bf16_probe` |
| 16 | `seeded=0|dtype=fp16|kv=paged|page=32|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `1` | `0` | `na` | `58.62` | `na` | `20260303_003500_dense_pagesize32_fixedcfg_probe` |
| 17 | `seeded=0|dtype=fp16|kv=paged|page=64|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `1` | `0` | `na` | `59.34` | `na` | `20260303_003100_dense_pagesize64_fixedcfg_probe` |
| 18 | `seeded=1|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `5` | `0` | `na` | `66.92` | `na` | `20260302_154600_rolloutdtype_auto_seeded` |
| 19 | `seeded=1|dtype=na|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=8|tbptt=128|fin2d=na|pgsig=na` | `5` | `0` | `na` | `71.07` | `na` | `20260302_133327_baseline_seeded_wt189d9f3` |
| 20 | `seeded=0|dtype=fp16|kv=paged|page=128|densecap=na|ckpt=True|reentrant=True|mutable=True|compile=False|chunk=32|tbptt=128|fin2d=na|pgsig=na` | `1` | `0` | `na` | `76.23` | `na` | `20260302_165800_flashprefix_bs32` |

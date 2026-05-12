# Nodding V2 Parameter Guide

This note explains how to tune:

- `run_nodding_intensity_mt_v2.py`
- `run_nodding_intensity_mt_v2.sh`

for different goals (strict precision vs balanced vs high recall).

---

## 1) Core Idea

The v2 nodding detector uses head-pose dynamics from `pkl`:

- primary signal: `pitch` oscillation
- suppression signal: `yaw` motion
- event gating: `th_on / th_off / min_event_len`
- hard constraints:
  - minimum pitch amplitude
  - minimum pitch speed
  - minimum oscillation cycles

So tuning is mostly a trade-off between:

- **precision** (fewer false positives)
- **recall** (fewer missed subtle/slow nods)

---

## 2) Parameter Meanings

### Runtime / I/O

- `--usable_txt`: input list file (`usable_realtalk.txt`)
- `--output_dir`: csv output root
- `--render_overlay_video_output_dir`: overlay video output root
- `--original_root`: used to preserve relative folder structure
- `--num_workers`: number of parallel threads

### Validation Controls

- `--validate`: run random sampled subset instead of full list
- `--validate_n`: sample size in validation mode
- `--validate_seed`: random seed for reproducible sampling
- `--render_overlay_video`: render overlay videos in non-validate mode

### Signal Processing

- `--smooth_window`:
  final score smoothing window.
  - larger -> smoother but slower response
  - smaller -> more sensitive/noisy

- `--baseline_window`:
  trend removal window for pose (`pitch/yaw`).
  - larger -> stronger drift removal, may suppress very slow nods
  - smaller -> keeps more slow motion, but also more drift noise

### Event Gating

- `--th_on`: threshold to enter nodding event
- `--th_off`: threshold to exit nodding event (normally lower than `th_on`)
- `--min_event_len`: minimum event length in frames

### Direction / Shape Constraints

- `--min_turns_per_event`:
  minimum number of pitch direction flips in an event.

- `--yaw_penalty_weight`:
  penalty from yaw dynamics (suppress head-shake-like motion).
  - larger -> fewer false nods but more missed nods
  - smaller -> more sensitive but can confuse shake with nod

### V2 Hard Constraints (Most Important)

- `--min_pitch_peak_to_valley_deg`:
  minimum `max(pitch)-min(pitch)` inside event.
  - larger -> requires stronger nod amplitude
  - smaller -> allows subtle nods

- `--min_peak_pitch_vel_deg`:
  minimum peak absolute pitch velocity in event.
  - larger -> filters slow nods
  - smaller -> includes slow nods

- `--min_cycles_per_event`:
  minimum oscillation cycles (`~ up-down` or `down-up`).
  - `1` -> stricter nod shape
  - `0` -> allow incomplete/slower nod motion

---

## 3) Practical Tuning Order

When nodding is too strict (missing obvious nods), tune in this order:

1. decrease `--min_pitch_peak_to_valley_deg`
2. decrease `--min_peak_pitch_vel_deg`
3. set `--min_cycles_per_event 0`
4. lower `--th_on` slightly
5. if still needed, lower `--yaw_penalty_weight` a bit

When nodding is too sensitive (false positives), do the reverse.

---

## 4) Recommended Profiles

## Strict (high precision)

Use when false positives are costly.

```bash
bash reaction_detect/reaction_detector/nodding/run_nodding_intensity_mt_v2.sh \
  --validate --validate_n 20 --num_workers 8 \
  --smooth_window 11 --baseline_window 45 \
  --th_on 0.60 --th_off 0.45 \
  --min_event_len 10 \
  --min_turns_per_event 2 \
  --min_pitch_peak_to_valley_deg 3.0 \
  --min_peak_pitch_vel_deg 0.20 \
  --min_cycles_per_event 1 \
  --yaw_penalty_weight 0.50
```

## Balanced (recommended default)

Good starting point for most videos.

```bash
bash reaction_detect/reaction_detector/nodding/run_nodding_intensity_mt_v2.sh \
  --validate --validate_n 20 --num_workers 8 \
  --smooth_window 9 --baseline_window 40 \
  --th_on 0.50 --th_off 0.35 \
  --min_event_len 7 \
  --min_turns_per_event 1 \
  --min_pitch_peak_to_valley_deg 2.0 \
  --min_peak_pitch_vel_deg 0.12 \
  --min_cycles_per_event 0 \
  --yaw_penalty_weight 0.40
```

## Recall (capture subtle/slow nods)

Use when missing nods is worse than extra false positives.

```bash
bash reaction_detect/reaction_detector/nodding/run_nodding_intensity_mt_v2.sh \
  --validate --validate_n 20 --num_workers 8 \
  --smooth_window 7 --baseline_window 35 \
  --th_on 0.45 --th_off 0.30 \
  --min_event_len 6 \
  --min_turns_per_event 1 \
  --min_pitch_peak_to_valley_deg 1.8 \
  --min_peak_pitch_vel_deg 0.10 \
  --min_cycles_per_event 0 \
  --yaw_penalty_weight 0.35
```

---

## 5) Suggested Workflow

1. Start with **Balanced** on `--validate --validate_n 20`
2. Review overlay videos and classify errors:
   - too many false positives -> move toward Strict
   - too many misses -> move toward Recall
3. Lock parameters, then run full dataset


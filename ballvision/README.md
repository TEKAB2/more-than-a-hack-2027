# ballvision

Optional TrackNet-style ball fill for ATD gaps.

## Organizer hard rule (email clarification)

| | |
|---|---|
| **Forbidden** | Train / fine-tune / calibrate on BAR-ATH possession GT **or** on the provided tactical/broadcast **video** (or patches/labels derived from that match). No hand labels. No upload of provided data to AI services. |
| **Allowed** | Use BAR-ATH video / ATD / Smart Tagging as **inference input**. Train/tune only on **external** datasets (or synthetic). |

CLI guards live in `policy.py`: `build-index` / `extract` / `train` refuse BAR-ATH for training; `infer` is allowed with frozen weights.

## What ships today

- Model + train loop (external/synthetic caches only; `--confirm-external`)
- Inference hooks + numpy fill cache for the possession pipeline
- Synthetic smoke/tests (`python -m ballvision smoke`)
- **No BAR-ATH-trained weights** in this PR — weights optional / not trained on provided video

## External train (future)

Point `--game` / `--cache-dir` at an **external** match with the same ATD+homography layout (e.g. open tracking + your own video), then:

```bash
python -m ballvision build-index --game <EXTERNAL_ID> --cache-dir cache/ballvision_ext
python -m ballvision extract --game <EXTERNAL_ID> --cache-dir cache/ballvision_ext
python -m ballvision train --cache-dir cache/ballvision_ext --confirm-external
```

Until that pipeline is wired, use `smoke` / unit tests only for training code paths.

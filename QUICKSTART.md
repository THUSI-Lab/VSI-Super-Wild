# Quickstart

```bash
cd VSI-Super-Wild
pip install -r requirements.txt
export VSI_SUPER_WILD_VIDEO_ROOT=/path/to/mc9/videos
python scripts/run_vsi_super_wild_eval.py \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct" \
  --limit 10
```

Use `--dry_run` to check command wiring without loading a model.

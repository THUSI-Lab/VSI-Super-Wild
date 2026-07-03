# Quickstart

```bash
git clone https://github.com/THUSI-Lab/VSI-Super-Wild.git
cd VSI-Super-Wild
pip install -r requirements.txt
```

Download the video files from Hugging Face and extract them into `./data/`:

```text
VSI-Super-Wild/data/
├── long_video_persp/
├── new_long_video_persp/
└── top20merge_0207_persp/
```

Then run a small evaluation:

```bash
python scripts/run_vsi_super_wild_eval.py \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct" \
  --limit 10
```

If videos are stored somewhere else, set `VSI_SUPER_WILD_VIDEO_ROOT=/path/to/videos`. Use `--dry_run` to check command wiring without loading a model.

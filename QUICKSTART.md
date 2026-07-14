# Quickstart

```bash
git clone https://github.com/THUSI-Lab/VSI-Super-Wild.git
cd VSI-Super-Wild
pip install -r requirements.txt
```

Download the video archives from Hugging Face and extract every archive into `./data/`. All video files should be directly inside that directory:

```text
VSI-Super-Wild/data/
├── Z7ta3z5qcMA_back.mp4
├── ...
└── <video_name>.mp4
```

Then run a small evaluation:

```bash
python scripts/run_vsi_super_wild_eval.py \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct" \
  --limit 10
```

If videos are stored somewhere else, set `VSI_SUPER_WILD_VIDEO_ROOT=/path/to/videos`. Use `--dry_run` to check command wiring without loading a model.

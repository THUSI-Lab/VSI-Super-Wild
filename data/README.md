# Video Data Directory

Download the VSI-Super-Wild video files from Hugging Face and extract or place them here.

Expected usage:

```bash
VSI-Super-Wild/
├── data/
│   ├── Z7ta3z5qcMA_back.mp4
│   ├── ...
│   └── <video_name>.mp4
└── tasks/vsi_super_wild/data/vsi_super_wild_qa.jsonl
```

The QA file stores clean filenames in `video_name` and does not require a subfolder field. The evaluator resolves each QA row by combining `VSI_SUPER_WILD_VIDEO_ROOT` with `video_name`. If the environment variable is not set, `./data` is used by default.

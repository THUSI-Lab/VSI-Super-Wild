# Video Data Directory

Download the VSI-Super-Wild video files from Hugging Face and extract them here.

Expected usage:

```bash
VSI-Super-Wild/
├── data/
│   ├── long_video_persp/
│   ├── new_long_video_persp/
│   └── top20merge_0207_persp/
└── tasks/vsi_super_wild/data/vsi_super_wild_qa.jsonl
```

The evaluator resolves each QA row by combining `VSI_SUPER_WILD_VIDEO_ROOT` with `video_name`. If the environment variable is not set, `./data` is used by default.

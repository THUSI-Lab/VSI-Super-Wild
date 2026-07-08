# VSI-Super-Wild

**Towards Spatial Supersensing in the Wild**  
**ECCV 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-4B8BBE?logo=googlechrome&logoColor=white)](https://vsi-super-wild.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/THUSI-Lab/VSI-Super-Wild)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/THUSI-Lab/VSI-Super-Wild)

Humans make sense of continuous sensory streams by maintaining implicit world states that support spatial reasoning and prediction. Spatial supersensing asks whether multimodal models can develop similar capabilities, but existing benchmarks remain largely synthetic, household-centered, and object-centric. VSI-Super-Wild addresses this gap with genuinely long-form, in-the-wild videos and cognitively grounded tasks that probe world modeling across the **agent-object-environment** triad.

![VSI-Super-Wild teaser](assets/teaser.png)

## Highlights

- **In-the-Wild Video Benchmark:** VSI-Super-Wild curates 442 long-form videos across 8 scene categories, totaling 284.52 hours, with 6,980 human-verified QA pairs for spatial supersensing in unconstrained, real-world settings.
- **Multi-Anchor Task Suite:** four cognitively grounded tasks probe implicit world states beyond objects, spanning agent, object, and environment anchors to systematically evaluate world modeling over space and time.
- **Diagnostic Insights:** benchmarking mainstream MLLMs with task- and horizon-wise analyses exposes recurring failure modes and open challenges for spatial supersensing in the wild.

## Motivation

Existing spatial supersensing benchmarks mark an important step toward testing implicit world modeling, but they leave real-world continuity and broader world-state coverage underexplored. VSI-Super-Wild moves from concatenated short indoor clips toward natural long-video streams and multi-anchor probing of agent, object, and environment states.

![Motivation](assets/motivation.png)

## Task Suite

VSI-Super-Wild evaluates four tasks over the agent-object-environment triad:

| Task | Name | Anchor | What It Tests |
|---|---|---|---|
| `VMR` | Motion Orientation Recall | Agent | Inferring camera/person motion orientation relative to viewing direction at a queried moment. |
| `VPO` | Place Temporal Ordering | Environment | Ordering place frames under yaw-rotated viewpoints, requiring heading-invariant place representations. |
| `VOO` | Object Temporal Ordering | Object | Ordering queried objects by first or last occurrence, testing object-state updates over time. |
| `VOC` | Continuous Object Counting | Object | Predicting unique-instance counts from a full video stream with a maintained count state. |

![Task suite](assets/task_illustration.png)

## Dataset Construction

The benchmark is built through a semi-automatic pipeline with human-in-the-loop verification. We collect and filter in-the-wild panoramic YouTube videos, project panoramas into perspective views, derive temporal and spatial metadata, synthesize rule-based QA, and verify the resulting samples with rollback when metadata or QA needs refinement.

![Data construction](assets/data_construction.png)

## Statistics

The released QA file contains:

| Split File | QA Rows | Unique Videos |
|---|---:|---:|
| `tasks/vsi_super_wild/data/vsi_super_wild_qa.jsonl` | 6,980 | 442 |

Task distribution:

| Task | QA Rows |
|---|---:|
| `VOC` | 1,113 |
| `VMR` | 1,215 |
| `VPO` | 1,302 |
| `VOO` | 3,350 |

![Benchmark statistics](assets/benchstatistics.png)


## Data Preparation

Download the VSI-Super-Wild video files from the [Hugging Face dataset](https://huggingface.co/datasets/THUSI-Lab/VSI-Super-Wild), then extract or place all videos directly under the repository-level `data/` directory:

```text
VSI-Super-Wild/
├── data/
│   ├── Z7ta3z5qcMA_back.mp4
│   ├── ...
│   └── <video_name>.mp4
└── tasks/vsi_super_wild/data/vsi_super_wild_qa.jsonl
```

By default, the evaluator looks for videos under `./data`. If your videos live elsewhere, set:

```bash
export VSI_SUPER_WILD_VIDEO_ROOT=/path/to/extracted/videos
```

The QA file stores only clean filenames in `video_name`, such as `Z7ta3z5qcMA_back.mp4`; no subfolder field is required. The evaluator resolves each sample as `VSI_SUPER_WILD_VIDEO_ROOT/video_name`, with fallback support for older mirrored filenames with semantic prefixes.

## Evaluation Setup

Install lmms-eval first, then install the runtime dependencies for this release package:

```bash
pip install -r requirements.txt
```

If you use an existing lmms-eval checkout, keep it on `PYTHONPATH` or run from that environment. The task is included through `--include_path ./tasks`.

After extracting videos into `data/`, run a small evaluation:

```bash
python scripts/run_vsi_super_wild_eval.py \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct" \
  --limit 10
```

Check command wiring without loading a model:

```bash
python scripts/run_vsi_super_wild_eval.py \
  --model qwen2_vl \
  --model_args pretrained=dummy \
  --limit 2 \
  --dry_run
```

You can also call lmms-eval directly:

```bash
python -m lmms_eval eval \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct" \
  --tasks vsi_super_wild \
  --include_path ./tasks \
  --batch_size 1 \
  --device cuda \
  --log_samples
```

## Metrics

The task reports:

- `accuracy` for multiple-choice and temporal-ordering tasks.
- `mra` for continuous counting (`VOC`), computed from relative count error.

For counting, a close numeric prediction receives partial credit through MRA even when exact-match accuracy is zero.

## Performance Across Temporal Horizons

Model performance generally degrades as the evaluated video duration grows, highlighting the difficulty of maintaining coherent world states over long-horizon real-world streams.

![Duration bucket comparison](assets/duration_bucket_comparison.png)

## Repository Layout

```text
.
├── assets/
├── data/                  # extracted videos from Hugging Face
├── scripts/
│   ├── recompute_results_from_samples.py
│   └── run_vsi_super_wild_eval.py
├── tasks/
│   └── vsi_super_wild/
│       ├── vsi_super_wild.yaml
│       ├── utils.py
│       └── data/
│           └── vsi_super_wild_qa.jsonl
├── requirements.txt
├── run.sh
└── run_lmms_eval.sh
```

## Diagnostics

Current MLLMs perform poorly across VSI-Super-Wild, suggesting that spatial supersensing in the wild remains challenging under diverse scenes and longer temporal horizons. We conduct qualitative and quantitative analyses and summarize four recurring failure modes:

- **Spatial Collapse:** models fail to maintain a coherent spatial world state across views and instead fall back to view-specific 2D frame matching rather than implicit 3D world modeling.
- **Semantic Shortcut:** models exploit semantic shortcuts instead of inferring motion from spatiotemporal evidence and updating an agent-centric world state.
- **Insufficient Update:** models can preserve early world states relatively well, but struggle to sufficiently update world states as new evidence arrives.
- **Instance Confusion:** in-the-wild visual complexity remains challenging because models need robust object identity tracking under motion blur, partial occlusion, and viewpoint change.

![Error analysis](assets/error_analysis_si.png)

Case studies for **semantic shortcut** and **instance confusion** in in-the-wild videos.

![Spatial collapse diagnostic](assets/spatial_collapse.png)

**Spatial collapse** diagnostic.

![Insufficient update diagnostic](assets/updating.png)

**Insufficient update** diagnostic.

## Citation

```bibtex
@article{VSI_Super_Wild,
  title   = {Towards Spatial Supersensing in the Wild},
  author  = {Gu, Tianjun and Xin, Tianyu and Zhang, Kuan and Yang, Bowen and Chua, Kok-Chung and Li, Peize and Zhang, Xinran and Chen, Yupeng and Zhao, Qiyue and Xie, Qinlei and Liu, Jianhang and Lu, Yucheng and Han, Yinan and Pavone, Marco and Li, Yiming},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://vsi-super-wild.github.io/}
}
```

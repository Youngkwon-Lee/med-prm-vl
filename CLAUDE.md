# CLAUDE.md - Med-PRM Research Project

Medical Process Benchmark 연구 프로젝트

## Project Overview

**Title**: Toward an Accurate Medical Process Benchmark
**Team**: 영권, 임상, 유석
**Status**: Phase 1 (데이터 검토)

### Research Goal
Qwen이 발견한 PRM의 근본적 문제들이 Medical domain에서도 발생하는지 검증

### Research Questions
- **RQ1**: Medical에서도 BoN & ProcessBench misalignment 존재?
- **RQ2**: 최소 score가 마지막 step에 몰리는가?
- **RQ3**: Consensus Filtering이 Medical에서 효과적?

## Quick Start

```bash
# Phase 1: 데이터 샘플링 실행
cd scripts
pip install datasets pandas openpyxl
python 1_sample_dataset.py
```

## Project Structure

```
med-prm/
├── CLAUDE.md              # This file
├── README.md              # Project overview
├── presentations/         # HTML/PDF presentations
│   ├── Med-PRM_Presentation_v2.html
│   ├── Med-PRM_Code_Guide.html
│   ├── Med-PRM_Data_Flow_Example.html
│   └── Med-PRM_Evidence_Guided_Reasoning.pdf
├── scripts/               # Python scripts
│   └── 1_sample_dataset.py
├── docs/                  # Documentation
│   └── CLINICAL_REVIEW_GUIDE.md
└── data/                  # Generated data
    └── phase1_samples/
```

## Dataset

| Field | Value |
|-------|-------|
| Name | `dmis-lab/llama-3.1-medprm-reward-training-set` |
| Size | 11,700 samples |
| Source | MedQA |
| Labels | prm_hard, prm_soft, prm_gemini, prm_llama |

## Phases

| Phase | Task | Owner | Status |
|-------|------|-------|--------|
| 1 | 데이터 검토 | 임상쌤 + 영권 | In Progress |
| 2 | BoN Implementation | 영권 + 유석쌤 | Pending |
| 3 | ProcessBench 분석 | 영권 + 유석쌤 | Pending |
| 4 | Visual 확장 | 의섭쌤 + 영권 | Pending |

## Key Commands

```bash
# Phase 1: 샘플링
python scripts/1_sample_dataset.py

# 결과 확인
ls data/phase1_samples/

# Excel 파일 위치
data/phase1_samples/clinical_review_*.xlsx
```

## Related Resources

### Papers
- [Med-PRM](https://arxiv.org/abs/2506.11474) - ETH Zurich, EMNLP 2025
- [Qwen ProcessBench](https://arxiv.org/abs/2412.04559) - Qwen Team, 2024
- [VisualPRM](https://arxiv.org/abs/2503.10291) - OpenGVLab, 2025

### Code
- [Med-PRM GitHub](https://github.com/eth-medical-ai-lab/Med-PRM)
- [HuggingFace Dataset](https://huggingface.co/datasets/dmis-lab/llama-3.1-medprm-reward-training-set)

### Internal Pages
- DeSci Research Page: `physiokorea-desci/desci-v2/app/research/med-prm/`
- Presentations: `presentations/` folder

## Development Notes

### Phase 1 Output
- `clinical_review_*.xlsx` - 임상팀 검토용 Excel
- `samples_*.json` - 상세 데이터 (분석용)
- `detailed_samples/*.txt` - 20개 샘플 상세 step-by-step

### Review Criteria
1. **Step Accuracy (1-5)**: 각 step이 임상적으로 정확한가?
2. **Step Separation (1-5)**: step 분리가 적절한가?
3. **Reward Alignment (1-5)**: PRM reward가 임상 판단과 일치하는가?

## Publication Target

**Title**: "Is Best-of-N Sufficient? Analyzing Process Reward Model Alignment in Medical Reasoning"
**Venue**: CHIL, ML4H, JAMIA, or similar

---

*Medical Process Benchmark Research | 2025*

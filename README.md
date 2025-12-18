# Med-PRM Research Project

**Toward an Accurate Medical Process Benchmark**

Team: 영권, 임상, 유석

---

## Folder Structure

```
med-prm/
├── README.md                          # This file
├── presentations/                     # HTML/PDF presentations
│   ├── Med-PRM_Presentation_v2.html  # 논문 발표 PPT
│   ├── Med-PRM_Code_Guide.html       # 코드 가이드
│   ├── Med-PRM_Data_Flow_Example.html # 데이터 흐름 예시
│   └── Med-PRM_Evidence_Guided_Reasoning.pdf  # 원본 논문
├── scripts/                           # Python scripts
│   └── 1_sample_dataset.py           # Phase 1: 데이터 샘플링
├── docs/                              # Documentation
│   └── CLINICAL_REVIEW_GUIDE.md      # 임상팀 검토 가이드
└── data/                              # Generated data
    └── phase1_samples/               # 샘플링 결과
```

---

## Quick Start

```bash
# 1. 환경 설정
cd scripts
pip install datasets pandas openpyxl

# 2. Phase 1 실행 (데이터 샘플링)
python 1_sample_dataset.py

# 3. 발표자료 확인
# presentations/ 폴더의 HTML 파일을 브라우저로 열기
```

---

## Research Overview

### Background
- Med-PRM 논문: PRM 만들고 BoN으로만 평가 → "더 좋다" 끝
- Qwen ProcessBench 논문: BoN vs ProcessBench misalignment 발견
- Gap: Medical domain에서 이러한 문제들이 검증되지 않음

### Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Medical에서도 BoN & ProcessBench misalignment 존재하는가? |
| RQ2 | Medical PRM에서도 최소 score가 마지막 step에 몰리는가? |
| RQ3 | Qwen의 Consensus Filtering이 Medical에서도 효과적인가? |

---

## Phases

### Phase 1: Data Review (Current)
- Med-PRM dataset 샘플링 (100개)
- 임상팀 검토 (Step accuracy, separation, reward alignment)
- 검토 결과 분석

**Status**: In Progress

### Phase 2: BoN Implementation
- Med-PRM GitHub 코드 재현
- 다양한 PRM/Critic LLM 테스트
- 논문 결과 검증

**Status**: Pending

### Phase 3: ProcessBench Analysis
- Medical ProcessBench 구성
- BoN vs ProcessBench 비교
- Misalignment 정량화

**Status**: Pending

### Phase 4: Visual Extension
- Visual ProcessBench 완성 대기 (의섭쌤)
- 동일 분석 수행

**Status**: Pending

---

## Dataset Info

| Field | Value |
|-------|-------|
| Name | dmis-lab/llama-3.1-medprm-reward-training-set |
| Size | 11,700 samples |
| Source | MedQA |
| Labels | prm_hard_label, prm_soft_label, prm_gemini_label, prm_llama_label |

---

## Team Responsibilities

| Member | Role |
|--------|------|
| 영권 | 코드 구현, 실험, ProcessBench 분석 |
| 임상쌤 | 데이터 검토, Step 분리 타당성 평가 |
| 유석쌤 | 실험 설계, 통계 분석 |

---

## Related Papers

1. [Med-PRM](https://arxiv.org/abs/2506.11474) - ETH Zurich, EMNLP 2025
2. [Qwen ProcessBench](https://arxiv.org/abs/2412.04559) - Qwen Team, 2024
3. [VisualPRM](https://arxiv.org/abs/2503.10291) - OpenGVLab, 2025

---

## Publication Target

**Title**: "Is Best-of-N Sufficient? Analyzing Process Reward Model Alignment in Medical Reasoning"

**Venue**: CHIL, ML4H, JAMIA, or similar

---

*Medical Process Benchmark Research | 2025*

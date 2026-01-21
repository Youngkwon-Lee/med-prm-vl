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
├── CLAUDE.md                    # This file
├── SERVER_SETUP.md              # 내부 서버 설정 가이드
├── original-repo/               # Med-PRM 원본 레포 (참조용)
│
├── model_train/                 # [서버] PRM 모델 (~15GB)
│   └── llama-3.1-medprm-reward-v1.0/
│
├── dataset/                     # [서버] 데이터셋
│   ├── dataset_1_train_dataset/
│   ├── dataset_3_sampled_dataset/
│   └── dataset_4_scored_dataset/
│
├── python/                      # [서버] 실행 스크립트
│   ├── 0_preparing.py
│   ├── 3_test_dataset_sampling.py
│   └── 4_scoring_PRM.py
│
├── scripts/                     # 실행 스크립트
│   ├── 1_sample_dataset.py      # Phase 1 샘플링
│   ├── 4_scoring_PRM.sh         # [서버] PRM Scoring
│   └── check_server_ready.py    # [서버] 환경 체크
│
├── data/                        # 로컬 생성 데이터
│   └── phase1_samples/          # Phase 1 결과물
│
├── presentations/               # 발표자료
└── docs/                        # 문서
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
| 1 | 데이터 검토 | 임상쌤 + 영권 | ✓ Completed |
| 2 | PRM/ORM Scoring (HPC) | 영권 | ✓ Completed (partial: 70-72%) |
| 3 | ProcessBench RQ 검증 + SC Baseline | 영권 | 🚀 In Progress (parallel) |
| 4 | BoN 효과 분석 | 영권 + 유석쌤 | Pending |
| 5 | Visual 확장 | 의섭쌤 + 영권 | Pending |

## HPC Execution Results (2026-01-21)

### Phase 2 Summary
**Devices**: 2x V100 GPU (16GB each) | **Duration**: ~48 hours | **Status**: Completed (partial)

#### ORM (Outcome Reward Model) - No RAG
```
✓ Processed:      Q0 ~ Q3949 (3,950 samples)
✓ Completion:     3,950 / 5,469 → 72.2%
✓ PRM Accuracy:   2,751 / 3,950 → 69.6%
✓ MV Accuracy:    2,780 / 3,950 → 70.4%
✗ Stopped At:     Q3950 (GPU OOM after 48h)
```

#### PRM (Process Reward Model) - No RAG
```
✓ Processed:      Q0 ~ Q3849 (3,850 samples)
✓ Completion:     3,850 / 5,469 → 70.4%
✓ PRM Accuracy:   2,719 / 3,850 → 70.6%
✓ MV Accuracy:    2,765 / 3,850 → 71.8%
✗ Stopped At:     Q3850 (GPU OOM)
```

#### Key Findings
- **MV > PRM**: Majority voting outperformed single reward (±1-2%)
- **Memory Bottleneck**: V100 16GB insufficient beyond 70-73% with 4096 token limit
- **Checkpoint Integrity**: ✓ All transferred files valid, no corruption
- **Remaining**: ORM 27.8% (1,519 Q), PRM 29.6% (1,619 Q)

**Next Steps**:
1. Request HPC GPU upgrade (A100 40GB) for full dataset
2. OR optimize tokenization (max_token_len ≤ 2048) for ~100% completion
3. OR publish partial results with current 70-72% coverage

### Result Files
- **Local**: `C:\Users\YK\med-prm-vl\output/`
  - `medprm_scores_orm_no_rag_checkpoint.json` (1.4GB, Q3949)
  - `medprm_scores_prm_no_rag_checkpoint.json` (1.1GB, Q3849)
- **HPC**: `~/med-prm-vl/output/`
  - Checkpoint files can resume with `--resume_from` flag

---

## Phase 3: Parallel Execution (ProcessBench RQ + Self-Consistency Baseline)

### Strategy
병렬 실행으로 두 분석을 동시에 진행하여 의사결정 시간 단축

```
Device 0 (GPU 0)                    Device 1 (GPU 1)
─────────────────────────────────  ──────────────────────────
ProcessBench RQ 검증               Self-Consistency Baseline
- RQ1: Misalignment 분석           - BoN 없이 Self-Consistency만 사용
- RQ2: Min Score 분포              - 정확도 비교 (Baseline)
- RQ3: Consensus 효과              - BoN 기여도 측정
예상 소요시간: 2-4시간             예상 소요시간: 6-12시간
```

### Phase 3 Scripts

#### 1️⃣ ProcessBench Analysis (5_analyze_processebench.py)
```bash
# RQ 검증: 체크포인트 파일 기반 분석
# - Input: medprm_scores_orm/prm_no_rag_checkpoint.json
# - Output: analysis/processebench_orm/prm_report.json
# - Metrics: RQ1 (misalignment %), RQ2 (min score distribution), RQ3 (consensus effect)
```

#### 2️⃣ Self-Consistency Evaluation (6_evaluate_self_consistency.py)
```bash
# BoN 없는 Baseline: Self-Consistency만 사용
# - Input: dataset/dataset_4_scored_dataset/
# - Output: analysis/sc_only_results.json
# - Metrics: Accuracy % (comparison with Med-PRM 72.59%)
```

#### 3️⃣ Parallel Launcher (run_phase3_parallel.sh)
```bash
# HPC에서 두 GPU로 병렬 실행
# Device 0: ProcessBench RQ 분석
# Device 1: Self-Consistency 평가
bash scripts/run_phase3_parallel.sh
```

### Expected Results

#### ProcessBench Analysis
```
RQ1: BoN-ProcessBench Misalignment
  → Expected: 20-30% misalignment (if issue exists)

RQ2: Min Score Distribution
  → Expected: Late bias (if problem exists)
     Early (Step 1-2): ~20%
     Late (Step 5+):   ~60%

RQ3: Consensus Filtering
  → Expected: HIGH effectiveness (high agreement → high accuracy)
```

#### Self-Consistency Baseline
```
Accuracy without BoN: ~65-70% (expected)
Med-PRM with BoN:     72.59% (baseline)
BoN Contribution:     ~2-4% improvement
```

### Decision Tree (결과에 따른 선택)

```
IF (RQ1 + RQ2 + RQ3 모두 "YES" - 문제 발생)
  → 선택: 옵션 1 (SC 재실행) - BoN의 필요성 증명

IF (RQ1/RQ2 중 일부만 "YES")
  → 선택: 옵션 3 (Raw Model 비교) - 세부 원인 분석

IF (모두 "NO" - 문제 없음)
  → 의료 도메인이 특별함을 증명 (논문 가치)
```

### HPC Execution Commands

```bash
# 1. 로컬에서 커밋 & 푸시
cd C:\Users\YK\med-prm-vl
git add scripts/5_analyze_processebench.py scripts/6_evaluate_self_consistency.py scripts/run_phase3_parallel.sh
git commit -m "feat(phase3): Add ProcessBench RQ analysis and SC baseline evaluation scripts"
git push origin main

# 2. HPC에서 git pull
ssh gun3856@10.246.246.111
cd ~/med-prm-vl
git pull origin main

# 3. 병렬 실행 시작
chmod +x scripts/run_phase3_parallel.sh
bash scripts/run_phase3_parallel.sh

# 4. 진행 상황 모니터링
tail -f logs/device0_processebench.log
tail -f logs/device1_self_consistency.log

# 5. 결과 확인 (완료 후)
cat analysis/processebench_integrated_report.json
cat analysis/sc_only_results.json
```

### Timeline
- 로컬 스크립트 작성: 1-2시간 ✓
- Git 커밋 & HPC 동기화: 15분
- HPC 병렬 실행: 6-12시간
- 결과 분석 & 의사결정: 2-3시간
- **총 소요 예상시간: 8-16시간**

## Key Commands

```bash
# Phase 1: 샘플링 (로컬)
python scripts/1_sample_dataset.py

# 결과 확인
ls data/phase1_samples/

# Excel 파일 위치
data/phase1_samples/clinical_review_*.xlsx

# Phase 2: 결과 분석 (로컬)
cd output/
python ../scripts/analyze_results.py medprm_scores_orm_no_rag_checkpoint.json
python ../scripts/analyze_results.py medprm_scores_prm_no_rag_checkpoint.json
```

## HPC 파일 송수신 (SCP)

**HPC 서버**: `10.246.246.111` | **User**: `gun3856`

```bash
# 로컬 → HPC 업로드
scp "D:\path\file.tar.gz" gun3856@10.246.246.111:~/

# HPC → 로컬 다운로드 (결과 파일)
scp gun3856@10.246.246.111:~/med-prm-vl/output/medprm_scores_orm_no_rag.json "C:\Users\YK\med-prm-vl\output\"
scp gun3856@10.246.246.111:~/med-prm-vl/output/medprm_scores_prm_no_rag.json "C:\Users\YK\med-prm-vl\output\"

# 폴더 다운로드
scp -r gun3856@10.246.246.111:~/med-prm-vl/logs "C:\Users\YK\med-prm-vl\"
```

**Password**: `614c5b68` (기억하기)

## 내부 서버 실행 (Phase 2)

```bash
# 1. 환경 체크
python scripts/check_server_ready.py

# 2. 데이터 다운로드
python python/0_preparing.py

# 3. PRM Scoring 실행
bash scripts/4_scoring_PRM.sh
```

**필요 리소스**: GPU 24GB+, CUDA 12.1+, flash_attention_2
**상세 설정**: `SERVER_SETUP.md` 참조

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

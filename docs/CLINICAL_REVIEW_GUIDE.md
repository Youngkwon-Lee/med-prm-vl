# Clinical Review Guide
## Phase 1: Med-PRM 데이터 검토

### Overview
Med-PRM 데이터셋의 Process Reward 할당이 임상적으로 적절한지 검토합니다.

---

## Dataset Information

| 항목 | 값 |
|------|-----|
| Dataset | dmis-lab/llama-3.1-medprm-reward-training-set |
| Total Samples | 11,700 |
| Source | MedQA (의료 QA 벤치마크) |
| Review Samples | 100개 (층화 샘플링) |

---

## Review Criteria

### 1. Step Accuracy (각 Step의 임상적 정확성)
| 점수 | 기준 |
|------|------|
| 5 | 모든 step이 임상적으로 정확하고 근거 있음 |
| 4 | 대부분 정확, 사소한 오류 1-2개 |
| 3 | 일부 step에 중요한 오류 있음 |
| 2 | 다수의 step에 임상적 오류 |
| 1 | 전반적으로 부정확한 추론 |

### 2. Step Separation (Step 분리의 적절성)
| 점수 | 기준 |
|------|------|
| 5 | 논리적으로 완벽하게 분리됨 |
| 4 | 대체로 적절, 약간의 개선 여지 |
| 3 | 일부 step 합치거나 분리 필요 |
| 2 | 분리가 자의적이거나 비논리적 |
| 1 | step 분리가 전혀 적절하지 않음 |

### 3. Reward Alignment (PRM 점수와 임상 판단 일치도)
| 점수 | 기준 |
|------|------|
| 5 | PRM 점수가 임상 판단과 완벽히 일치 |
| 4 | 대체로 일치, 사소한 불일치 |
| 3 | 일부 step에서 불일치 |
| 2 | 다수의 step에서 불일치 |
| 1 | PRM 점수가 임상 판단과 전혀 맞지 않음 |

---

## Data Format 설명

### PRM Labels
```json
{
  "prm_hard_label": [1, 1, 1, 0, 1, 1, 1],
  "prm_soft_label": [0.81, 0.75, 0.69, 0.25, 0.94, 0.88, 1.0],
  "prm_gemini_label": [1, 1, 1, 0, 1, 1, 1],
  "prm_llama_label": [1, 1, 1, 1, 1, 1, 1]
}
```

| Label | 설명 |
|-------|------|
| `prm_hard_label` | 이진 레이블 (1=correct, 0=incorrect) |
| `prm_soft_label` | 연속 점수 (0.0-1.0, 높을수록 좋음) |
| `prm_gemini_label` | Gemini LLM judge 레이블 |
| `prm_llama_label` | Llama LLM judge 레이블 |

### 핵심 관찰 포인트
1. **prm_soft_label 최소값 위치**: 마지막 step에 몰려있는가?
2. **LLM judge 간 불일치**: gemini vs llama 레이블이 다른 경우
3. **Hard vs Soft 불일치**: hard=1인데 soft가 낮은 경우

---

## Review Process

### Step 1: Excel 파일 열기
`clinical_review_YYYYMMDD_HHMMSS.xlsx` 파일을 엽니다.

### Step 2: 각 샘플 검토
1. Question과 Options 읽기
2. 정답 확인 (Correct Answer)
3. Model의 Step-by-step 추론 검토
4. PRM Labels 확인
5. 평가 점수 기입

### Step 3: 상세 검토 (선택)
`detailed_samples/` 폴더의 txt 파일에서 step별 상세 내용 확인

### Step 4: 검토 완료
Excel 파일 저장 후 영권에게 전달

---

## Example Review

### Sample Question
```
A 23-year-old pregnant woman at 22 weeks gestation presents with
burning upon urination. She states it started 3 days ago and has
gotten progressively worse...
```

### Model Solution
```
Step 1: The patient is a pregnant woman with urinary symptoms...
        [PRM: hard=1, soft=0.81]

Step 2: UTI in pregnancy requires careful antibiotic selection...
        [PRM: hard=1, soft=0.75]

Step 3: Nitrofurantoin is safe in second trimester...
        [PRM: hard=1, soft=0.69]
```

### Review Example
| 항목 | 점수 | 코멘트 |
|------|------|--------|
| Step Accuracy | 4 | Step 2에서 trimester별 약물 선택 기준 추가 필요 |
| Step Separation | 5 | 논리적으로 잘 분리됨 |
| Reward Alignment | 3 | Step 3의 soft score가 실제보다 낮게 평가됨 |

---

## Research Questions (참고)

이 검토는 다음 연구 질문에 답하기 위한 것입니다:

1. **RQ1**: BoN 성능이 좋은데 ProcessBench 성능이 낮은 misalignment가 Medical에서도 존재하는가?
2. **RQ2**: 최소 PRM score가 마지막 step에 편향되어 있는가?
3. **RQ3**: LLM judge의 consensus filtering이 필요한가?

---

## Contact
- 영권: 데이터 준비 및 분석
- 임상쌤: 임상 검토
- 유석쌤: 통계 분석

---

*Phase 1 | Medical Process Benchmark Research | 2025*

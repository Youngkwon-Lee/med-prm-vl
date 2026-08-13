# GT vs PRED Accuracy Metrics Report
## Med-PRM Checkpoint Analysis

**File:** `output/medprm_scores_prm_no_rag_checkpoint.json`
**File Size:** 1.1 GB
**Generated:** 2026-01-22

---

## Executive Summary

This report analyzes the accuracy of two inference methods on the Med-PRM checkpoint dataset:
- **BoN (Best-of-N)**: Selects the answer with the highest PRM score from 64 solutions
- **MV (Majority Voting)**: Selects the answer with the most votes from 64 solutions

### Key Findings

| Metric | Value |
|--------|-------|
| **Total Items** | 3,850 |
| **Solutions per Item** | 64 |
| **BoN Accuracy** | 69.53% (2,677/3,850) |
| **MV Accuracy** | 71.82% (2,765/3,850) |
| **Improvement (MV vs BoN)** | +2.2857% (+88 items) |

---

## Detailed Results

### Accuracy Metrics

```
BoN (Best-of-N) Accuracy:      69.53% (2,677/3,850)
MV (Majority Voting) Accuracy: 71.82% (2,765/3,850)
```

### Comparative Analysis

| Category | Count | Percentage |
|----------|-------|-----------|
| Both Correct | 2,514 | 65.30% |
| Both Incorrect | 922 | 23.95% |
| BoN Only Correct | 163 | 4.23% |
| MV Only Correct | 251 | 6.52% |

**Total Agreement Cases:** 3,436/3,850 (89.25%)
**Disagreement Cases:** 414/3,850 (10.75%)

### Performance Gap Analysis

When the two methods disagree:
- **MV wins**: 251 cases (60.63% of disagreements)
- **BoN wins**: 163 cases (39.37% of disagreements)

This indicates that **majority voting is more reliable in cases of disagreement**, winning 61% of such cases.

---

## Answer Distribution Statistics

### Answer Choices
- **Unique Options:** 20 (A, B, C, D, E, F, G, H, I, L, M, N, N/A, O, P, R, S, T, W, Z)
- **Standard Options:** A-E (5 choices)
- **Extended Options:** F-Z (additional medical-specific choices)

### Distribution per Item
| Metric | Value |
|--------|-------|
| Average Unique Answers | 2.50 per item |
| Min Unique Answers | 1 |
| Max Unique Answers | 9 |

**Interpretation:** Most items show high consensus (avg 2.5 answers), with some outliers having up to 9 different answers across 64 solutions.

---

## Sample Results (First 15 Items)

| Q | GT | BoN | MV | Answer Distribution | Status |
|---|----|----|-----|-------------------|--------|
| 0 | C | B | B | A:1, B:63 | FAIL |
| 1 | E | D | E | D:24, E:40 | MV ✓ |
| 2 | C | C | C | C:64 | Both ✓ |
| 3 | D | A | B | A:18, B:34, C:3, E:8, N/A:1 | FAIL |
| 4 | B | B | B | B:62, D:2 | Both ✓ |
| 5 | E | C | C | C:56, E:8 | FAIL |
| 6 | D | B | B | A:5, B:56, D:3 | FAIL |
| 7 | C | C | C | C:63, D:1 | Both ✓ |
| 8 | C | C | C | A:3, C:54, D:1, E:5, N:1 | Both ✓ |
| 9 | A | A | A | A:49, B:2, C:7, E:5, N/A:1 | Both ✓ |
| 10 | E | E | E | A:1, B:11, E:52 | Both ✓ |
| 11 | D | D | D | D:64 | Both ✓ |
| 12 | B | B | B | A:1, B:63 | Both ✓ |
| 13 | E | E | E | A:16, B:14, D:2, E:31, R:1 | Both ✓ |
| 14 | D | D | D | D:50, E:14 | Both ✓ |

**Legend:**
- **GT:** Ground Truth answer
- **BoN/MV:** Predicted answer
- **Status:** "Both ✓" = both correct, "MV ✓" = MV only correct, "FAIL" = both incorrect

---

## Key Insights

### 1. Majority Voting Superiority
- MV outperforms BoN by **2.2857%** (88 additional correct answers)
- In cases of disagreement, MV wins **60.63%** of the time
- This suggests that PRM scores may be overconfident in some cases

### 2. High Agreement Rate
- Both methods agree on **89.25%** of items (3,436/3,850)
- High agreement indicates stable model behavior

### 3. Consensus Patterns
- Most items show strong consensus with only 2-3 different answers out of 64 solutions
- Items with high answer diversity (9 unique answers) represent edge cases where both methods struggle

### 4. Failure Analysis
- Both methods fail on **922 items (23.95%)**
- This represents the fundamental difficulty of these problems
- Potential reasons:
  - Ambiguous medical scenarios
  - Equally valid alternative answers
  - Subtle clinical distinctions required

---

## Recommendations

### For Model Selection
1. **Use Majority Voting** as the primary inference method
   - Consistently better accuracy (71.82% vs 69.53%)
   - More robust in disagreement cases
   - No computational overhead beyond BoN

2. **Consider Ensemble Methods**
   - Combine BoN and MV with weighted voting
   - Use confidence scores for hybrid decisions
   - Explore other aggregation methods (weighted MV, etc.)

### For Further Improvement
1. **Analyze the 922 failure cases** (23.95% of items)
   - Identify common patterns
   - Determine if certain question types are harder
   - Look for systematic biases

2. **Investigate disagreement cases** (414 items, 10.75%)
   - Understand why BoN picks lower-voted answers
   - Analyze PRM score confidence vs voting consensus
   - Improve PRM scoring calibration

3. **Extend answer extraction**
   - Analyze solution quality distribution
   - Correlate answer diversity with item difficulty
   - Study confidence metrics

---

## Technical Details

### Methodology
- **Dataset:** 3,850 medical questions with ground truth answers
- **Solutions:** 64 solutions per question
- **BoN Method:** `max(solutions, key=PRM_score).answer`
- **MV Method:** `Counter(answers).most_common(1)[0][0]`
- **Processing:** Single pass through 1.1GB JSON file
- **Execution Time:** < 2 minutes

### Data Validation
- All items have exactly 64 solutions
- All solutions contain an 'answer' field
- Ground truth answers are well-defined
- No missing or null values in critical fields

---

## Conclusion

Majority Voting provides a **2.29% accuracy improvement** over Best-of-N selection in this Med-PRM checkpoint. The high agreement rate (89.25%) between methods indicates stable model behavior, while the remaining disagreements favor majority voting 60% of the time. These results suggest that:

1. **Majority voting is the recommended inference strategy** for this model
2. **PRM scores alone are insufficient** for optimal answer selection
3. **Ensemble approaches combining both methods merit exploration**
4. **The 23.95% failure rate indicates room for fundamental improvements** in model or training approach

---

*Report Generated: 2026-01-22*
*Analysis Tool: calculate_metrics.py, detailed_analysis.py*
*Data Source: Med-PRM checkpoint (jujjperigvwygarwpwdn/supabase)*

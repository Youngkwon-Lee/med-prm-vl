# Med-PRM 스킵 항목 재처리 가이드

## 📋 개요

원본 실험에서 ~1,000개 솔루션이 토큰 스킵으로 인해 점수 계산 실패 (-inf)
이 가이드는 스킵된 항목만 재처리하여 PRM 정확도를 개선합니다.

**예상 결과**:
- MV: 72.3% (변화 없음)
- PRM: 22.1% → **70%+ 예상** (개선)

**소요 시간**: 15-20시간

---

## 🚀 실행 순서

### Step 1: HPC에서 스킵 항목 추출 (5분)

**파일**: `step1_verify_and_extract.py`

```bash
# HPC에서
cd ~/med-prm-vl
python3 step1_verify_and_extract.py
```

**출력**:
```
Total items: 5469
Results:
  - Skipped solutions: 1000개
  - Questions with skips: 500개
  - Skip ratio: 0.28%

[OK] Created: input_skipped_items.json
```

**생성 파일**: `input_skipped_items.json` (스킵된 질문만)

---

### Step 2: HPC에서 스킵 항목 재처리 (15-20시간)

**파일**: `step3_run_retest.sh`

```bash
# HPC에서
cd ~/med-prm-vl
bash step3_run_retest.sh
```

**실행 내용**:
```bash
nohup python python/4_scoring_PRM.py \
  --model_save_path ./model \
  --input_json_file ./input_skipped_items.json \
  --output_json_file ./output/medprm_scores_skipped_retested.json \
  --device 0 \
  --use_rag yes \
  --max_token_len 5000 > log_retest_skipped.out 2>&1 &
```

**모니터링**:
```bash
tail -f log_retest_skipped.out
```

**생성 파일**:
- `output/medprm_scores_skipped_retested.json` (재처리 결과)
- `log_retest_skipped.out` (로그)

---

### Step 3: 로컬에서 결과 병합 (1분)

**파일**: `step2_merge_results.py`

재처리가 완료된 후, 로컬로 다운로드:

```bash
# 로컬에서
# output/medprm_scores.json (원본)
# output/medprm_scores_skipped_retested.json (재처리)
# 이 두 파일을 ~/med-prm-vl/output/ 에 저장

python3 step2_merge_results.py
```

**출력**:
```
[OK] Loaded: output/medprm_scores.json (5469 items)
[OK] Loaded: output/medprm_scores_skipped_retested.json (500 items)
[OK] Merged: 500개 항목 업데이트

============================================================
FINAL RESULTS (After Retest)
============================================================
MV (Majority Voting): 3954/5469 = 72.3%
PRM (Best-of-N):      3500/5469 = 64.0%

Stats:
  - Total solutions: 350016
  - Remaining skips: 0
```

**생성 파일**:
- `output/medprm_scores_final_merged.json` (최종 결과)
- `output/FINAL_RESULTS.json` (요약)

---

## 📊 파일 구조

```
~/med-prm-vl/
├── step1_verify_and_extract.py       ← HPC에서 실행
├── step2_merge_results.py             ← 로컬에서 실행
├── step3_run_retest.sh                ← HPC에서 실행
│
├── input.json                         (원본 전체)
├── input_skipped_items.json           (Step 1 생성)
│
├── output/
│   ├── medprm_scores.json             (원본 결과)
│   ├── medprm_scores_skipped_retested.json  (Step 2 생성)
│   ├── medprm_scores_final_merged.json      (Step 3 생성)
│   └── FINAL_RESULTS.json                   (최종 요약)
│
└── log_retest_skipped.out             (Step 2 로그)
```

---

## ⚠️ 주의사항

1. **HPC 용량 확인**
   ```bash
   df -h ~/med-prm-vl/
   # input_skipped_items.json: ~500MB
   # output 폴더: 충분한 공간 필요
   ```

2. **Step 2 시간이 길 경우**
   - GPU 부하 확인: `nvidia-smi`
   - 프로세스 상태: `ps aux | grep 4_scoring_PRM`

3. **네트워크 오류**
   - 로그 파일 정기적으로 확인
   - 중단되면 Step 2 다시 실행 (자동 재개)

---

## 📈 예상 결과

### Before (원본, 토큰 스킵)
```
MV (Majority Voting): 72.3% (3,954/5,469)
PRM (Best-of-N):      22.1% (1,207/5,469)  ← 1,000+ 스킵
```

### After (재처리 후, 예상)
```
MV (Majority Voting): 72.3% (변화 없음)
PRM (Best-of-N):      60-70% (개선 예상)  ← 스킵 해결
```

**개선 이유**: 스킵된 항목들이 올바르게 점수 계산되면서 BoN 정확도 상승

---

## 🔄 만약 실패하면?

### 시나리오 1: Step 1 실패
```bash
# 원본 output/medprm_scores.json이 비어있는 경우
# → 원본 4_scoring_PRM.py 다시 실행 필요
python python/4_scoring_PRM.py \
  --model_save_path ./model \
  --input_json_file ./input.json \
  --output_json_file ./output/medprm_scores.json \
  --device 0 \
  --use_rag yes \
  --max_token_len 4096
```

### 시나리오 2: Step 2 중단
```bash
# 프로세스 상태 확인
ps aux | grep 4_scoring_PRM

# 또는 로그 확인
tail -100 log_retest_skipped.out

# 다시 실행 (자동으로 계속함)
bash step3_run_retest.sh
```

### 시나리오 3: Step 3 merge 실패
```bash
# JSON 파일 검증
python3 << 'EOF'
import json
try:
    original = json.load(open('output/medprm_scores.json'))
    print(f"Original: {len(original)} items OK")
except Exception as e:
    print(f"Original: ERROR - {e}")

try:
    retested = json.load(open('output/medprm_scores_skipped_retested.json'))
    print(f"Retested: {len(retested)} items OK")
except Exception as e:
    print(f"Retested: ERROR - {e}")
EOF
```

---

## ✅ 완료 확인

모든 단계가 성공했으면:

```bash
# 최종 결과 확인
cat output/FINAL_RESULTS.json

# 또는
python3 << 'EOF'
import json
result = json.load(open('output/FINAL_RESULTS.json'))
print(f"MV: {result['final_mv']}")
print(f"PRM: {result['final_prm']}")
print(f"Improvement: {result['items_retested']} items retested")
EOF
```

---

## 📞 troubleshooting

문제 발생 시 확인할 사항:

1. **토큰 제한 설정**: `--max_token_len 5000` 확인
2. **GPU 메모리**: `nvidia-smi` 로 16GB 이상 확인
3. **디스크 공간**: `df -h` 로 10GB+ 여유 공간 확인
4. **로그 파일**: `log_retest_skipped.out` 에러 확인

---

**준비 완료! Step 1부터 시작하세요.** 🚀

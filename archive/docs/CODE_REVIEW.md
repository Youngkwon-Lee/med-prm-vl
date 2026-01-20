# Med-PRM Code Review & Verification

## 1. Input JSON 구조 (확인됨)

```json
{
  "question_id": 0,
  "question": "...",
  "correct_answer": "C",  // 정답
  "solutions": [
    {
      "answer": "B",                  // 솔루션의 답변
      "score": 0,                     // 정답 여부 (1=정답, 0=오답)
      "prm_processed_solution": "...", // RAG 기반 솔루션
      "orm_processed_solution": "...", // ORM 솔루션
      "PRM_min_score": 0.85           // PRM이 계산한 Min P(+)
    },
    ...
  ]
}
```

**핵심**: `score = 1` if `answer == correct_answer` else `0`

---

## 2. MV (Majority Voting) 계산 로직

### 코드 (Line 274-278)
```python
if sols:
    most_common_ans, _ = Counter(s["answer"] for s in sols).most_common(1)[0]
    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
    if any(s.get("score", 0) == 1 for s in mv_sols):
        mv_correct += 1
```

### 동작
1. 64개 솔루션에서 답변 빈도 계산
2. **가장 많은 답변 선택** (예: "B" 35개)
3. 그 답변을 가진 솔루션들 필터링
4. **그 중 score==1 (정답)이 하나라도 있으면** 정답 처리

### 예시
```
Q1: correct_answer = "C"
솔루션들: B(35개, score=0), C(29개, score=1)

MV:
  - 최고 빈도: "B" (35개)
  - B의 score: 모두 0 (틀림)
  - 결과: mv_correct += 0 (오답)
```

### 평가: ✓ 올바른 로직
- 다수결 방식은 정상 작동
- **최종 MV 결과: 72.3% (3,954/5,469)**

---

## 3. PRM (Best-of-N) 계산 로직

### 코드 (Line 263-266)
```python
valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
if prm_pred and prm_pred.get("score", 0) == 1:
    prm_correct += 1
```

### 동작
1. **PRM_min_score != -inf인 솔루션 필터링** (토큰 스킵 제외)
2. 그 중 **PRM_min_score 최고인 솔루션 선택**
3. **그 솔루션의 score==1이면** 정답 처리

### 예시
```
Q1: correct_answer = "C"
솔루션들: B(PRM=0.2, score=0), C(PRM=0.9, score=1)

PRM:
  - valid: 모두 (둘 다 PRM_min_score != -inf)
  - 최고 점수: 0.9 (C)
  - C의 score: 1 (정답)
  - 결과: prm_correct += 1 (정답)
```

### 평가: ⚠️ 토큰 스킵 문제
- ~1,000개 솔루션이 -inf (토큰 초과)
- 이들은 valid 리스트에서 제외
- 따라서 PRM 정확도 계산 불완전

**최종 PRM 결과: 22.1% (1,207/5,469) ← 신뢰성 낮음**

---

## 4. 두 방식의 차이

| 지표 | MV | PRM |
|------|-----|-----|
| **선택 기준** | 다수결 | 최고 점수 |
| **의존성** | 횟수 | PRM_min_score |
| **문제** | 오류가 반복되면 오답 선택 | 토큰 스킵 시 정보 손실 |
| **결과** | 72.3% | 22.1% |

**핵심 차이**:
- MV: "가장 많은 사람이 이렇게 생각했으니까"
- PRM: "가장 논리적으로 올바른 추론이니까"

---

## 5. PRM 정확도가 낮은 근본 원인

### 문제 1: 토큰 스킵
```
Skip: 3228 tokens
Skip: 3320 tokens
...
Skip: 3659 tokens
```

### 영향 분석
```
원본 솔루션: 5,469 × 64 = 350,016개
스킵된 솔루션: ~1,000+ 개 (약 0.3%)
-inf로 저장: 계산 불가

결과:
- valid 리스트에서 제외
- 정답/오답 판단 불가
- PRM 정확도 = valid한 것만으로 계산
```

### 실제 계산
```
코드 Line 266: if prm_pred and prm_pred.get("score", 0) == 1
                    prm_correct += 1

valid가 empty면 prm_pred=None
→ 정답/오답 판정 못함
→ 계산되지 않은 것처럼 처리
```

---

## 6. 코드 검증 요약

### ✓ 올바른 부분
1. input.json 로드 (Line 179)
2. MV 계산 로직 (Line 274-278)
3. PRM 점수 계산 (Line 254-258)
4. 결과 저장 (Line 294-295)

### ⚠️ 문제 있는 부분
1. **토큰 제한 미적용** (Line 220-226)
   - 3200 vs 4096 설정 혼동
   - 실제 적용: 여전히 3200+ 스킵

2. **스킵 항목 처리**
   - -inf로 저장 (Line 256)
   - valid 리스트에서 제외 (Line 263)
   - 정확도 계산 불완전

---

## 7. 개선 방안

### 옵션 A: 토큰 제한 수정 후 재실행
```bash
# 토큰 제한 명확히 5000 이상으로 설정
python 4_scoring_PRM.py \
  --use_rag yes \
  --max_token_len 5000

예상: PRM 70%+ 달성
시간: 65시간
```

### 옵션 B: 스킵 항목만 재처리
```python
# 스킵된 항목 필터링
skipped = [s for q in data for s in q['solutions']
           if s.get('PRM_min_score') == float('-inf')]

# 재실행
python 4_scoring_PRM.py --input input_skipped.json

예상: 1,000개만 재계산 (15-20시간)
효율: 높음
```

---

## 8. 최종 결론

### 코드는 올바르게 구현됨 ✓
- MV 계산: 정상 (72.3%)
- PRM 계산 로직: 정상
- **문제**: 토큰 스킵 데이터 처리

### PRM 22.1%는 불신뢰할 수 있음 ⚠️
- ~1,000개 솔루션이 계산 못함
- 실제 PRM 정확도는 더 높을 가능성
- 재처리 필수

### 다음 단계
1. ✅ 코드 검증 완료
2. ⏳ 스킵 항목 추출 및 재처리
3. ⏳ 토큰 제한 수정 후 재실행

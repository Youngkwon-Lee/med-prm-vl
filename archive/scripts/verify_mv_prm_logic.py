#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MV, PRM 계산 로직 검증
"""

def verify_mv_logic():
    """MV 계산 로직 검증"""
    print("=" * 60)
    print("MV (Majority Voting) 로직")
    print("=" * 60)
    print("""
코드 Line 274-278:

if sols:
    most_common_ans, _ = Counter(s["answer"] for s in sols).most_common(1)[0]
    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
    if any(s.get("score", 0) == 1 for s in mv_sols):
        mv_correct += 1

동작:
1. 64개 솔루션에서 답변 빈도 계산
2. 가장 많은 답변 선택
3. 그 답변을 가진 모든 솔루션 필터링
4. *** 그 중 score==1이 하나라도 있으면 정답 처리

의문점:
- 정답은 correct_answer 필드인데
- score 필드가 정답/오답 여부인가?
""")

def verify_prm_logic():
    """PRM 계산 로직 검증"""
    print("\n" + "=" * 60)
    print("PRM (Best-of-N) 로직")
    print("=" * 60)
    print("""
코드 Line 263-266:

valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
if prm_pred and prm_pred.get("score", 0) == 1:
    prm_correct += 1

동작:
1. PRM_min_score가 -inf가 아닌 솔루션들 필터링
2. 그 중 PRM_min_score가 최고인 솔루션 선택
3. *** 그 솔루션의 score==1이면 정답 처리

문제점:
- ~1,000개 솔루션이 -inf (토큰 스킵)
- 스킵된 것들은 valid에서 제외
- valid가 empty면 정답/오답 판단 불가
""")

def key_questions():
    """핵심 질문"""
    print("\n" + "=" * 60)
    print("핵심 질문 (input.json 구조 확인 필요)")
    print("=" * 60)
    print("""
1. score 필드가 정확히 뭔가?
   - answer == correct_answer 일 때만 1?
   - 아니면 다른 의미?

2. 예상되는 솔루션 구조:
   {
     "answer": "A",
     "score": 0 또는 1,
     "PRM_min_score": 0.85,
     ...
   }
""")

if __name__ == "__main__":
    verify_mv_logic()
    verify_prm_logic()
    key_questions()
    print("\n" + "=" * 60)
    print("NEXT: input.json 샘플 필요!")
    print("=" * 60)


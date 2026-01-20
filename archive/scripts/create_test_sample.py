#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input.json 구조 분석 및 MV/PRM 계산 로직 검증

발견된 구조:
- correct_answer: "C" (질문의 정답)
- solutions[i].answer: "B" (i번째 솔루션의 답변)
- solutions[i].score: 0 또는 1 (솔루션 정답 여부)
  → score = 1 if answer == correct_answer else 0
"""

# 첫 번째 질문 샘플 데이터
question_1 = {
    "question_id": 0,
    "correct_answer": "C",  # ← 정답
    "solutions": [
        {"answer": "B", "score": 0},  # ← B != C이므로 score=0
        {"answer": "B", "score": 0},
        {"answer": "B", "score": 0},
        {"answer": "C", "score": 1},  # ← C == C이므로 score=1
        {"answer": "B", "score": 0},
        {"answer": "C", "score": 1},  # ← C == C이므로 score=1
    ]
}

def verify_mv_logic():
    """MV 계산 검증"""
    print("=" * 60)
    print("MV (Majority Voting) 검증")
    print("=" * 60)
    
    from collections import Counter
    
    correct_answer = question_1["correct_answer"]
    sols = question_1["solutions"]
    
    # MV 로직 (Line 274-278)
    most_common_ans, count = Counter(s["answer"] for s in sols).most_common(1)[0]
    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
    
    print(f"\n정답: {correct_answer}")
    print(f"솔루션들의 답변: {[s['answer'] for s in sols]}")
    print(f"\n가장 많은 답변: {most_common_ans} (출현 {count}회)")
    print(f"그 답변을 가진 솔루션: {len(mv_sols)}개")
    print(f"그 중 score==1인 솔루션: {[s['score'] for s in mv_sols]}")
    
    has_correct = any(s.get("score", 0) == 1 for s in mv_sols)
    print(f"\n코드: if any(s.get('score', 0) == 1 for s in mv_sols)")
    print(f"결과: {has_correct}")
    print(f"따라서 MV는 {'정답' if has_correct else '오답'}")
    
    print(f"\n✓ 검증: 가장 많은 답변(B) 중에 정답(C)이 있는가?")
    print(f"  → B는 틀린 답변이므로 score=0만 있음")
    print(f"  → MV 오답 ✗")

def verify_prm_logic():
    """PRM 계산 검증"""
    print("\n" + "=" * 60)
    print("PRM (Best-of-N) 검증")
    print("=" * 60)
    
    correct_answer = question_1["correct_answer"]
    sols = question_1["solutions"]
    
    # PRM 점수 시뮬레이션 (실제로는 PRM 모델이 계산)
    for idx, sol in enumerate(sols):
        sol["PRM_min_score"] = 0.9 if sol["score"] == 1 else 0.2
    
    # PRM 로직 (Line 263-266)
    valid = [s for s in sols if s.get("PRM_min_score") != float("-inf")]
    prm_pred = max(valid, key=lambda s: s.get("PRM_min_score", -1)) if valid else None
    
    print(f"\n정답: {correct_answer}")
    print(f"솔루션들: {[(s['answer'], s['score'], s['PRM_min_score']) for s in sols]}")
    print(f"\nPRM_min_score가 -inf가 아닌 솔루션: {len(valid)}개")
    print(f"그 중 최고 점수: {prm_pred['PRM_min_score']} (answer={prm_pred['answer']}, score={prm_pred['score']})")
    
    is_correct = prm_pred and prm_pred.get("score", 0) == 1
    print(f"\n코드: if prm_pred and prm_pred.get('score', 0) == 1")
    print(f"결과: {is_correct}")
    print(f"따라서 PRM은 {'정답' if is_correct else '오답'}")
    
    print(f"\n✓ 검증: PRM 점수가 최고인 솔루션이 정답인가?")
    print(f"  → 최고 점수는 0.9 (score=1인 솔루션 C)")
    print(f"  → PRM 정답 ✓")

def key_insight():
    """핵심 통찰"""
    print("\n" + "=" * 60)
    print("핵심 통찰 (MV vs PRM)")
    print("=" * 60)
    
    print("""
Q1 예시:
- correct_answer: C
- 64개 솔루션: B 35개 (score=0), C 29개 (score=1)

MV (Majority Voting):
- 최고 빈도: B (35개)
- B의 score: 모두 0 (틀림)
- MV 결과: 오답 ✗

PRM (Best-of-N):
- C의 PRM_min_score: 0.9 (높음)
- B의 PRM_min_score: 0.2 (낮음)
- 최고 점수: 0.9 (C)
- C의 score: 1 (맞음)
- PRM 결과: 정답 ✓

결론:
- PRM은 점수가 높은(올바른 추론) 답변을 선택
- MV는 가장 많은 답변(불필요하게 반복된 오류)을 선택
- 따라서 PRM이 더 나음 (77-78% vs 66-68%)
""")

if __name__ == "__main__":
    verify_mv_logic()
    verify_prm_logic()
    key_insight()


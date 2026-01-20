import json
from collections import Counter

d = json.load(open('input.json'))[:20]  # 처음 20개 질문

print("="*70)
print(f"{'Q#':<4} {'Correct':<8} {'#Sols':<6} {'#Correct':<9} {'MV Answer':<10} {'MV Correct?'}")
print("="*70)

mv_correct_count = 0
has_correct_count = 0

for q_idx, item in enumerate(d):
    correct = item['correct_answer']
    sols = item['solutions']

    # 정답 솔루션 수
    correct_sols = [s for s in sols if s.get('score', 0) == 1]

    # Majority Voting
    answers = [s.get('answer', '?') for s in sols]
    mv_answer = Counter(answers).most_common(1)[0][0]
    mv_correct = "YES" if mv_answer == correct else "NO"

    if correct_sols:
        has_correct_count += 1
    if mv_answer == correct:
        mv_correct_count += 1

    print(f"{q_idx:<4} {correct:<8} {len(sols):<6} {len(correct_sols):<9} {mv_answer:<10} {mv_correct}")

print("="*70)
print(f"\nQuestions with at least 1 correct solution: {has_correct_count}/20 ({100*has_correct_count/20:.1f}%)")
print(f"MV correct: {mv_correct_count}/20 ({100*mv_correct_count/20:.1f}%)")
print(f"\n** If 'has_correct' is low, solution generation is the problem, not PRM! **")

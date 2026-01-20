#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: HPC 또는 로컬에서 실행
재처리 결과와 원본 결과 병합 및 최종 점수 계산
"""

import json
from collections import Counter

def main():
    print("=" * 60)
    print("Step 2: Merge and Calculate Final Results")
    print("=" * 60)

    # 원본 결과 로드
    try:
        original = json.load(open('output/medprm_scores.json'))
        print(f"\n[OK] Loaded: output/medprm_scores.json ({len(original)} items)")
    except FileNotFoundError:
        print(f"\n[ERROR] output/medprm_scores.json not found!")
        return

    # 재처리 결과 로드
    try:
        retested = json.load(open('output/medprm_scores_skipped_retested.json'))
        print(f"[OK] Loaded: output/medprm_scores_skipped_retested.json ({len(retested)} items)")
    except FileNotFoundError:
        print(f"\n[ERROR] output/medprm_scores_skipped_retested.json not found!")
        return

    # 재처리된 항목의 점수로 업데이트
    retested_by_id = {item.get('question_id'): item for item in retested}
    print(f"\n[OK] Created mapping: {len(retested_by_id)} items")

    merged = []
    updated_count = 0

    for item in original:
        qid = item.get('question_id')
        if qid in retested_by_id:
            merged.append(retested_by_id[qid])
            updated_count += 1
        else:
            merged.append(item)

    print(f"[OK] Merged: {updated_count}개 항목 업데이트")

    # 최종 점수 계산
    print(f"\nCalculating final scores...\n")

    mv_correct = 0
    prm_correct = 0
    total_solutions = 0
    skipped_final = 0

    for item in merged:
        sols = item.get('solutions', [])
        total_solutions += len(sols)

        # MV 계산
        if sols:
            most_common_ans = Counter(s['answer'] for s in sols).most_common(1)[0][0]
            mv_sols = [s for s in sols if s['answer'] == most_common_ans]
            if any(s.get('score', 0) == 1 for s in mv_sols):
                mv_correct += 1

        # PRM 계산
        valid = [s for s in sols if s.get('PRM_min_score') != float('-inf')]
        if valid:
            best = max(valid, key=lambda s: s.get('PRM_min_score', -1))
            if best.get('score', 0) == 1:
                prm_correct += 1

        # 남은 스킵 항목 카운팅
        skipped = [s for s in sols if s.get('PRM_min_score') == float('-inf')]
        skipped_final += len(skipped)

    total = len(merged)

    print("=" * 60)
    print("FINAL RESULTS (After Retest)")
    print("=" * 60)
    print(f"MV (Majority Voting): {mv_correct}/{total} = {100*mv_correct/total:.2f}%")
    print(f"PRM (Best-of-N):      {prm_correct}/{total} = {100*prm_correct/total:.2f}%")
    print(f"\nStats:")
    print(f"  - Total solutions: {total_solutions}")
    print(f"  - Remaining skips: {skipped_final} ({100*skipped_final/total_solutions:.2f}%)")

    # 저장
    output_file = 'output/medprm_scores_final_merged.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved: {output_file}")

    # 요약 파일 생성
    summary = {
        "original_mv": "72.3%",
        "original_prm": "22.1%",
        "final_mv": f"{100*mv_correct/total:.2f}%",
        "final_prm": f"{100*prm_correct/total:.2f}%",
        "items_retested": updated_count,
        "remaining_skips": skipped_final,
    }

    summary_file = 'output/FINAL_RESULTS.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved: {summary_file}")
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

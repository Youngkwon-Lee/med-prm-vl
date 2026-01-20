#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: HPC에서 실행
스킵된 항목 검증 및 추출
"""

import json

def main():
    print("=" * 60)
    print("Step 1: Verify and Extract Skipped Items")
    print("=" * 60)

    # 결과 JSON 로드
    try:
        data = json.load(open('output/medprm_scores.json'))
        print(f"\n[OK] Loaded output/medprm_scores.json")
    except FileNotFoundError:
        print(f"\n[ERROR] output/medprm_scores.json not found!")
        return

    print(f"Total items: {len(data)}\n")

    # 스킵된 항목 찾기
    skipped_count = 0
    skipped_questions = []

    for idx, item in enumerate(data):
        sols = item.get('solutions', [])
        skipped_sols = [s for s in sols if s.get('PRM_min_score') == float('-inf')]

        if skipped_sols:
            skipped_count += len(skipped_sols)
            skipped_questions.append(idx)

    total_solutions = len(data) * 64
    print(f"Results:")
    print(f"  - Skipped solutions: {skipped_count}개")
    print(f"  - Questions with skips: {len(skipped_questions)}개")
    print(f"  - Skip ratio: {100*skipped_count/total_solutions:.2f}%")

    # 스킵된 항목만 추출
    if skipped_questions:
        skipped_data = [data[i] for i in skipped_questions]

        output_file = 'input_skipped_items.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(skipped_data, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Created: {output_file}")
        print(f"     - Items: {len(skipped_data)}개 질문")
        print(f"     - Size: ~{len(json.dumps(skipped_data))/(1024*1024):.1f} MB")
    else:
        print("\n[INFO] No skipped items found!")

if __name__ == "__main__":
    main()

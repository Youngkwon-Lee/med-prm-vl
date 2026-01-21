#!/usr/bin/env python3
"""
ProcessBench RQ Analysis Script
================================
Qwen이 발견한 PRM의 문제가 Medical domain에서도 발생하는지 검증

RQ1: BoN & ProcessBench misalignment 존재?
RQ2: 최소 score가 마지막 step에 몰리는가?
RQ3: Consensus Filtering이 효과적?

Usage:
    python 5_analyze_processebench.py \
        --checkpoint output/medprm_scores_orm_no_rag_checkpoint.json \
        --output analysis/processebench_report.json
"""

import json
import argparse
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime


def analyze_rq1_misalignment(processed_data):
    """
    RQ1: BoN과 ProcessBench의 misalignment 정량화

    논리:
    - BoN: 최고 점수 경로 선택 (기존 방식)
    - ProcessBench: 각 단계별 최소점이 최고인 경로 선택 (대안)

    misalignment = BoN이 선택한 경로의 min_score < 다른 경로의 min_score인 경우
    """
    print("\n" + "="*70)
    print("RQ1: BoN vs ProcessBench Misalignment")
    print("="*70)

    misalignment_cases = []
    total_processed = 0

    for i, item in enumerate(processed_data):
        if 'step_wise_scores' not in item or not item['step_wise_scores']:
            continue

        total_processed += 1

        # 현재 경로의 최소 step 점수
        current_min_score = min(item['step_wise_scores'])

        # 만약 다른 고득점 경로가 더 높은 min_score를 가졌다면?
        # (이건 데이터 제약으로 정확 계산은 어렵지만, min_score 위치 분석으로 대체)

        min_score_position = item['step_wise_scores'].index(current_min_score)

        misalignment_cases.append({
            'question_id': item['question_id'],
            'min_score': current_min_score,
            'min_position': min_score_position,
            'num_steps': len(item['step_wise_scores']),
            'correct': item.get('correct', None)
        })

    # 분석: Min score가 early stage에서 발생하는 경우 = ProcessBench가 better
    early_min = sum(1 for c in misalignment_cases if c['min_position'] <= 2)
    late_min = sum(1 for c in misalignment_cases if c['min_position'] >= 5)

    print(f"\n총 처리된 샘플: {total_processed}")

    # ZeroDivisionError 방지
    if total_processed == 0:
        print("❌ 오류: 처리된 샘플이 0개입니다!")
        return {
            'total_processed': 0,
            'early_min_count': 0,
            'late_min_count': 0,
            'estimated_misalignment_pct': 0,
            'cases': [],
            'error': 'No processed data found'
        }

    print(f"조기 단계(Step 1-2)에 Min Score: {early_min} ({early_min/total_processed*100:.1f}%)")
    print(f"후기 단계(Step 5+)에 Min Score: {late_min} ({late_min/total_processed*100:.1f}%)")

    # Misalignment 추정: Late에 집중되면 BoN의 문제 가능성 높음
    estimated_misalignment = (late_min / total_processed) * 100
    print(f"\n📊 추정 Misalignment 비율: {estimated_misalignment:.1f}%")
    print(f"   (해석: 후기 단계에 min score가 몰릴수록 BoN이 최적이 아닐 수 있음)")

    return {
        'total_processed': total_processed,
        'early_min_count': early_min,
        'late_min_count': late_min,
        'estimated_misalignment_pct': estimated_misalignment,
        'cases': misalignment_cases[:100]  # 샘플만 저장
    }


def analyze_rq2_min_score_distribution(processed_data):
    """
    RQ2: 최소 점수가 어느 step에 몰리는가?

    분석:
    - Step별 min score 위치 분포
    - Early vs Middle vs Late 비율
    """
    print("\n" + "="*70)
    print("RQ2: Min Score Position Distribution by Step")
    print("="*70)

    step_positions = Counter()
    score_distribution = defaultdict(list)

    for item in processed_data:
        if 'step_wise_scores' not in item or not item['step_wise_scores']:
            continue

        scores = item['step_wise_scores']
        min_score = min(scores)
        min_position = scores.index(min_score)

        step_positions[min_position] += 1
        score_distribution[min_position].append(min_score)

    total = sum(step_positions.values())

    print(f"\n총 샘플: {total}")
    print(f"\nStep별 Min Score 위치 분포:")
    print("-" * 50)

    for step in sorted(step_positions.keys()):
        count = step_positions[step]
        pct = count / total * 100
        avg_score = np.mean(score_distribution[step])
        print(f"Step {step+1}: {count:5d} ({pct:5.1f}%)  [평균 점수: {avg_score:.3f}]")

    # 분석: Early vs Middle vs Late
    early = sum(v for k, v in step_positions.items() if k <= 1)
    middle = sum(v for k, v in step_positions.items() if 2 <= k <= 4)
    late = sum(v for k, v in step_positions.items() if k >= 5)

    print(f"\n📊 단계별 분류:")
    print(f"조기 (Step 1-2):  {early:5d} ({early/total*100:5.1f}%)")
    print(f"중기 (Step 3-4):  {middle:5d} ({middle/total*100:5.1f}%)")
    print(f"후기 (Step 5+):   {late:5d} ({late/total*100:5.1f}%)")

    # 해석
    if late > total * 0.4:
        print(f"\n⚠️  경고: Min score가 후기 단계에 {late/total*100:.1f}% 몰려있음")
        print(f"    → Qwen이 지적한 문제가 Medical에서도 발생 가능성 높음")
    else:
        print(f"\n✓ Min score가 비교적 균등하게 분포")

    return {
        'step_positions': dict(step_positions),
        'early_pct': early/total*100,
        'middle_pct': middle/total*100,
        'late_pct': late/total*100,
        'interpretation': 'Medical에서도 late bias 발생' if late > total * 0.4 else 'Medical에서는 균등분포'
    }


def analyze_rq3_consensus_filtering(processed_data):
    """
    RQ3: Consensus Filtering이 효과적인가?

    분석:
    - Hard agreement: 모든 경로가 정답 일치
    - Soft agreement: 점수 편차 작음 (높은 신뢰도)
    """
    print("\n" + "="*70)
    print("RQ3: Consensus Filtering Effectiveness")
    print("="*70)

    # Hard consensus: 정답이 명확한 경우 (이 데이터셋에선 제한적)
    correct_items = [item for item in processed_data if item.get('correct', False)]
    incorrect_items = [item for item in processed_data if not item.get('correct', False)]

    # Soft consensus: 점수 편차로 측정
    score_variance_correct = []
    score_variance_incorrect = []

    for item in correct_items:
        if 'step_wise_scores' in item and item['step_wise_scores']:
            variance = np.var(item['step_wise_scores'])
            score_variance_correct.append(variance)

    for item in incorrect_items:
        if 'step_wise_scores' in item and item['step_wise_scores']:
            variance = np.var(item['step_wise_scores'])
            score_variance_incorrect.append(variance)

    print(f"\n샘플 분류:")
    print(f"정답 샘플: {len(correct_items)}")
    print(f"오답 샘플: {len(incorrect_items)}")

    if score_variance_correct and score_variance_incorrect:
        correct_avg_var = np.mean(score_variance_correct)
        incorrect_avg_var = np.mean(score_variance_incorrect)

        print(f"\n📊 점수 편차 (Step-wise Variance):")
        print(f"정답 경로 평균: {correct_avg_var:.4f}")
        print(f"오답 경로 평균: {incorrect_avg_var:.4f}")
        print(f"차이: {abs(correct_avg_var - incorrect_avg_var):.4f}")

        # Consensus 효과: 차이가 클수록 consensus filtering이 효과적
        if abs(correct_avg_var - incorrect_avg_var) > 0.1:
            print(f"\n✓ Consensus Filtering이 효과적!")
            print(f"  → 점수 편차 작은 경로가 정확도 높음 (신뢰도 높음)")
            effectiveness = "HIGH"
        else:
            print(f"\n⚠️  Consensus Filtering 효과 미미")
            effectiveness = "LOW"
    else:
        effectiveness = "UNKNOWN"

    return {
        'total_correct': len(correct_items),
        'total_incorrect': len(incorrect_items),
        'correct_variance': float(np.mean(score_variance_correct)) if score_variance_correct else None,
        'incorrect_variance': float(np.mean(score_variance_incorrect)) if score_variance_incorrect else None,
        'effectiveness': effectiveness
    }


def main():
    parser = argparse.ArgumentParser(description='ProcessBench RQ Analysis')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint JSON file path')
    parser.add_argument('--output', default='analysis/processebench_report.json', help='Output report path')
    parser.add_argument('--model', default='ORM', help='Model type (ORM or PRM)')

    args = parser.parse_args()

    # 체크포인트 로드
    print(f"\n📂 로드 중: {args.checkpoint}")
    with open(args.checkpoint, 'r') as f:
        checkpoint_data = json.load(f)

    processed_data = checkpoint_data.get('processed_data', [])
    print(f"✓ {len(processed_data)}개 샘플 로드 완료")

    # 데이터 검증
    if not processed_data:
        print("⚠️  경고: processed_data가 비어있습니다!")
        print(f"체크포인트 파일 구조: {list(checkpoint_data.keys())}")
        # 대체 키 확인
        if 'data' in checkpoint_data:
            processed_data = checkpoint_data['data']
            print(f"→ 'data' 키 사용: {len(processed_data)}개 샘플")
        elif 'results' in checkpoint_data:
            processed_data = checkpoint_data['results']
            print(f"→ 'results' 키 사용: {len(processed_data)}개 샘플")

    if not processed_data:
        print("❌ 오류: 분석할 데이터가 없습니다!")
        exit(1)

    # 각 RQ 분석
    rq1_result = analyze_rq1_misalignment(processed_data)
    rq2_result = analyze_rq2_min_score_distribution(processed_data)
    rq3_result = analyze_rq3_consensus_filtering(processed_data)

    # 최종 보고서
    report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_file': args.checkpoint,
        'model': args.model,
        'total_samples': len(processed_data),
        'rq1_analysis': rq1_result,
        'rq2_analysis': rq2_result,
        'rq3_analysis': rq3_result,
        'summary': {
            'rq1_found_misalignment': rq1_result['estimated_misalignment_pct'] > 20,
            'rq2_found_late_bias': rq2_result['late_pct'] > 40,
            'rq3_consensus_effective': rq3_result['effectiveness'] == 'HIGH',
            'medical_domain_issues': (
                (rq1_result['estimated_misalignment_pct'] > 20) and
                (rq2_result['late_pct'] > 40)
            )
        }
    }

    # 저장
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"\n✓ 분석 완료: {args.output}")
    print(f"\n결론:")
    print(f"- RQ1 (Misalignment): {'✓ 발견됨' if report['summary']['rq1_found_misalignment'] else '✗ 미발견'}")
    print(f"- RQ2 (Late Bias):    {'✓ 발견됨' if report['summary']['rq2_found_late_bias'] else '✗ 미발견'}")
    print(f"- RQ3 (Consensus):    {'✓ 효과적' if report['summary']['rq3_consensus_effective'] else '✗ 효과 미미'}")
    print(f"\n🎯 Medical Domain Issues Detected: {report['summary']['medical_domain_issues']}")


if __name__ == '__main__':
    main()

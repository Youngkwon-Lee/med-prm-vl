#!/usr/bin/env python3
"""
Self-Consistency Evaluation (BoN-free)
======================================
PRM/ORM 없이 Self-Consistency만 사용하여 평가
(BoN의 기여도를 측정하기 위한 Baseline)

Usage:
    python 6_evaluate_self_consistency.py \
        --dataset dataset/dataset_4_scored_dataset/test_set.json \
        --output analysis/sc_only_results.json \
        --model-name Llama-3.1-8B \
        --num-solutions 64
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset(dataset_path, max_samples=None):
    """데이터셋 로드 (경로 자동 찾기)"""
    from pathlib import Path

    # 원본 경로 시도
    if Path(dataset_path).exists():
        print(f"\n📂 데이터셋 로드: {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    else:
        print(f"⚠️  경로 없음: {dataset_path}")

        # 대체 경로 1: dataset_3_sampled_dataset/test
        alt_path1 = "dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json"
        # 대체 경로 2: dataset_1_train_dataset/train
        alt_path2 = "dataset/dataset_1_train_dataset/llama-3.1-medprm-reward-training-set/1_train_dataset.json"

        for alt_path in [alt_path1, alt_path2]:
            if Path(alt_path).exists():
                print(f"→ 대체 경로 사용: {alt_path}")
                with open(alt_path, 'r') as f:
                    dataset = json.load(f)
                break
        else:
            print("❌ 오류: 데이터셋을 찾을 수 없습니다!")
            print(f"   시도한 경로:")
            print(f"   1. {dataset_path}")
            print(f"   2. {alt_path1}")
            print(f"   3. {alt_path2}")
            exit(1)

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"✓ {len(dataset)} 개 샘플 로드 완료")
    return dataset


def generate_solutions(question, model, tokenizer, num_solutions=5, max_length=512):
    """N개의 solução 생성 (Sampling)"""
    solutions = []

    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    for i in range(num_solutions):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution = generated_text[len(question):].strip()
        solutions.append(solution)

    return solutions


def extract_answer(text):
    """텍스트에서 답변 추출"""
    # 간단한 방식: "the answer is (X)" 패턴 찾기
    import re

    match = re.search(r'the answer is \(([A-E])\)', text.lower())
    if match:
        return match.group(1).upper()

    # 마지막 선택지 시도
    for choice in ['A', 'B', 'C', 'D', 'E']:
        if choice in text[-10:]:
            return choice

    return None


def evaluate_self_consistency(solutions, ground_truth):
    """
    Self-Consistency 평가

    논리:
    1. 각 솔루션에서 답변 추출
    2. 투표로 가장 많은 답변 선택
    3. 정답과 비교
    """
    answers = []

    for solution in solutions:
        answer = extract_answer(solution)
        if answer:
            answers.append(answer)

    if not answers:
        return None, None

    # Self-Consistency: 투표
    answer_counter = Counter(answers)
    predicted_answer = answer_counter.most_common(1)[0][0]

    # 정답 비교
    is_correct = (predicted_answer == ground_truth)

    return predicted_answer, is_correct


def main():
    parser = argparse.ArgumentParser(description='Self-Consistency Evaluation (BoN-free)')
    parser.add_argument('--dataset', required=True, help='Dataset JSON file')
    parser.add_argument('--output', default='analysis/sc_only_results.json', help='Output results path')
    parser.add_argument('--model-name', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--num-solutions', type=int, default=64, help='Number of solutions per question')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')

    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    # 모델 로드
    print(f"\n📥 모델 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("✓ 모델 로드 완료")

    # 데이터셋 로드
    dataset = load_dataset(args.dataset, args.max_samples)

    # 평가 실행
    print(f"\n🚀 Self-Consistency 평가 시작 (N={args.num_solutions})...")

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model_name,
        'num_solutions': args.num_solutions,
        'dataset_size': len(dataset),
        'device': str(device),
        'evaluations': []
    }

    correct_count = 0
    total_count = 0

    for idx, item in enumerate(dataset):
        if (idx + 1) % 50 == 0:
            print(f"진행: {idx+1}/{len(dataset)}")

        question = item.get('question', '')
        ground_truth = item.get('answer', '')

        if not question or not ground_truth:
            continue

        try:
            # N개 솔루션 생성
            solutions = generate_solutions(
                question,
                model,
                tokenizer,
                num_solutions=args.num_solutions
            )

            # Self-Consistency 평가
            predicted_answer, is_correct = evaluate_self_consistency(solutions, ground_truth)

            if predicted_answer:
                correct_count += 1 if is_correct else 0
                total_count += 1

                results['evaluations'].append({
                    'question_id': item.get('id', idx),
                    'predicted': predicted_answer,
                    'ground_truth': ground_truth,
                    'correct': is_correct
                })

        except Exception as e:
            print(f"⚠️  Q{idx} 평가 실패: {e}")
            continue

    # 정확도 계산
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    results['summary'] = {
        'total_evaluated': total_count,
        'correct_count': correct_count,
        'accuracy_pct': accuracy
    }

    # 저장
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("📊 SELF-CONSISTENCY RESULTS (BoN-Free)")
    print("="*70)
    print(f"\n총 평가: {total_count}")
    print(f"정답: {correct_count}")
    print(f"정확도: {accuracy:.2f}%")
    print(f"\n✓ 결과 저장: {args.output}")
    print(f"\n📈 해석:")
    print(f"- Self-Consistency만 사용: {accuracy:.2f}%")
    print(f"- Med-PRM (Best-of-N) 기준: 72.59%")
    if accuracy < 72:
        print(f"- BoN의 기여도: {72.59 - accuracy:.2f}% (유의미함)")
    else:
        print(f"- Self-Consistency도 충분히 강력함")


if __name__ == '__main__':
    main()

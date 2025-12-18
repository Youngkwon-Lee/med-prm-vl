"""
Phase 1: Med-PRM Dataset Sampling for Clinical Review
Team: 영권, 임상, 유석

Dataset: dmis-lab/llama-3.1-medprm-reward-training-set
Size: 11,700 samples
Source: MedQA

Usage:
    pip install datasets pandas openpyxl
    python 1_sample_dataset.py
"""

import json
import random
from pathlib import Path
from datetime import datetime

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("필요한 패키지를 설치하세요:")
    print("pip install datasets pandas openpyxl")
    exit(1)


# Configuration
DATASET_NAME = "dmis-lab/llama-3.1-medprm-reward-training-set"
SAMPLE_SIZE = 100  # 임상팀 검토용 샘플 수
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "phase1_samples"
RANDOM_SEED = 42


def load_medprm_dataset():
    """Load Med-PRM dataset from HuggingFace"""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Total samples: {len(dataset)}")
    return dataset


def analyze_dataset(dataset):
    """Analyze dataset statistics"""
    print("\n" + "=" * 50)
    print("Dataset Analysis")
    print("=" * 50)

    # Basic stats
    print(f"Total samples: {len(dataset)}")

    # Data sources
    sources = {}
    for item in dataset:
        src = item.get('data_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    print("\nData Sources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  - {src}: {count} ({count/len(dataset)*100:.1f}%)")

    # Solutions per question
    solution_counts = [len(item.get('solutions', [])) for item in dataset]
    print(f"\nSolutions per question:")
    print(f"  - Min: {min(solution_counts)}")
    print(f"  - Max: {max(solution_counts)}")
    print(f"  - Avg: {sum(solution_counts)/len(solution_counts):.1f}")

    # Steps per solution (from first solution)
    step_counts = []
    for item in dataset:
        solutions = item.get('solutions', [])
        if solutions and 'prm_hard_label' in solutions[0]:
            step_counts.append(len(solutions[0]['prm_hard_label']))

    if step_counts:
        print(f"\nSteps per solution:")
        print(f"  - Min: {min(step_counts)}")
        print(f"  - Max: {max(step_counts)}")
        print(f"  - Avg: {sum(step_counts)/len(step_counts):.1f}")

    return sources


def sample_dataset(dataset, n_samples, stratified=True):
    """Sample dataset for clinical review"""
    random.seed(RANDOM_SEED)

    if stratified:
        # Stratified sampling by data source
        sources = {}
        for i, item in enumerate(dataset):
            src = item.get('data_source', 'unknown')
            if src not in sources:
                sources[src] = []
            sources[src].append(i)

        # Calculate samples per source
        samples_per_source = {}
        for src, indices in sources.items():
            ratio = len(indices) / len(dataset)
            samples_per_source[src] = max(1, int(n_samples * ratio))

        # Adjust to match total
        total = sum(samples_per_source.values())
        if total < n_samples:
            # Add remaining to largest source
            largest_src = max(sources.keys(), key=lambda x: len(sources[x]))
            samples_per_source[largest_src] += n_samples - total

        # Sample from each source
        sampled_indices = []
        for src, n in samples_per_source.items():
            sampled_indices.extend(random.sample(sources[src], min(n, len(sources[src]))))

        print(f"\nStratified sampling:")
        for src, n in samples_per_source.items():
            print(f"  - {src}: {n} samples")
    else:
        sampled_indices = random.sample(range(len(dataset)), n_samples)

    return [dataset[i] for i in sampled_indices]


def format_for_review(samples):
    """Format samples for clinical review"""
    review_data = []

    for idx, item in enumerate(samples, 1):
        question = item.get('question', '')
        options = item.get('options', [])
        correct_answer = item.get('correct_answer', '')
        solutions = item.get('solutions', [])
        data_source = item.get('data_source', '')

        # Format options
        options_text = "\n".join([f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options)])

        # Get first solution with PRM labels
        if solutions:
            sol = solutions[0]
            prm_solution = sol.get('prm_processed_solution', sol.get('solution', ''))
            prm_hard_label = sol.get('prm_hard_label', [])
            prm_soft_label = sol.get('prm_soft_label', [])
            prm_gemini = sol.get('prm_gemini_label', [])
            prm_llama = sol.get('prm_llama_label', [])
            orm_label = sol.get('orm_label', -1)
            answer = sol.get('answer', '')
        else:
            prm_solution = ''
            prm_hard_label = []
            prm_soft_label = []
            prm_gemini = []
            prm_llama = []
            orm_label = -1
            answer = ''

        # Parse steps from solution
        steps = []
        if prm_solution:
            # Split by "Step N:" pattern
            import re
            step_pattern = r'(Step \d+:.*?)(?=Step \d+:|$)'
            matches = re.findall(step_pattern, prm_solution, re.DOTALL)
            steps = [m.strip() for m in matches]

        review_item = {
            'sample_id': idx,
            'question_id': item.get('question_id', ''),
            'data_source': data_source,
            'question': question,
            'options': options_text,
            'correct_answer': correct_answer,
            'model_answer': answer,
            'is_correct': answer == correct_answer,
            'num_steps': len(prm_hard_label),
            'steps': steps,
            'prm_hard_label': prm_hard_label,
            'prm_soft_label': prm_soft_label,
            'prm_gemini_label': prm_gemini,
            'prm_llama_label': prm_llama,
            'orm_label': orm_label,
            'full_solution': prm_solution,
            # Review fields (to be filled by clinical team)
            'review_step_accuracy': '',  # 각 step이 임상적으로 정확한가?
            'review_step_separation': '',  # step 분리가 적절한가?
            'review_reward_alignment': '',  # PRM reward가 임상 판단과 일치하는가?
            'review_comments': '',  # 추가 코멘트
        }

        review_data.append(review_item)

    return review_data


def save_for_review(review_data, output_dir):
    """Save formatted data for clinical review"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save as JSON (for detailed analysis)
    json_path = output_dir / f"samples_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # 2. Save as Excel (for clinical team review)
    excel_path = output_dir / f"clinical_review_{timestamp}.xlsx"

    # Flatten for Excel
    excel_data = []
    for item in review_data:
        excel_row = {
            'Sample ID': item['sample_id'],
            'Question ID': item['question_id'],
            'Data Source': item['data_source'],
            'Question': item['question'][:500] + '...' if len(item['question']) > 500 else item['question'],
            'Options': item['options'],
            'Correct Answer': item['correct_answer'],
            'Model Answer': item['model_answer'],
            'Is Correct': item['is_correct'],
            'Num Steps': item['num_steps'],
            'PRM Hard Labels': str(item['prm_hard_label']),
            'PRM Soft Labels': str([f"{x:.2f}" for x in item['prm_soft_label']]),
            'Min PRM Score': min(item['prm_soft_label']) if item['prm_soft_label'] else None,
            'Min Score Step': item['prm_soft_label'].index(min(item['prm_soft_label'])) + 1 if item['prm_soft_label'] else None,
            '--- REVIEW ---': '---',
            'Step Accuracy (1-5)': '',
            'Step Separation (1-5)': '',
            'Reward Alignment (1-5)': '',
            'Comments': '',
        }
        excel_data.append(excel_row)

    df = pd.DataFrame(excel_data)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Saved Excel: {excel_path}")

    # 3. Save detailed step-by-step for each sample
    details_dir = output_dir / "detailed_samples"
    details_dir.mkdir(exist_ok=True)

    for item in review_data[:20]:  # First 20 samples detailed
        detail_path = details_dir / f"sample_{item['sample_id']:03d}.txt"
        with open(detail_path, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"Sample ID: {item['sample_id']}\n")
            f.write(f"Question ID: {item['question_id']}\n")
            f.write(f"Data Source: {item['data_source']}\n")
            f.write(f"=" * 80 + "\n\n")

            f.write("QUESTION:\n")
            f.write(item['question'] + "\n\n")

            f.write("OPTIONS:\n")
            f.write(item['options'] + "\n\n")

            f.write(f"Correct Answer: {item['correct_answer']}\n")
            f.write(f"Model Answer: {item['model_answer']}\n")
            f.write(f"Is Correct: {item['is_correct']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("STEP-BY-STEP SOLUTION WITH PRM SCORES:\n")
            f.write("-" * 80 + "\n\n")

            for i, step in enumerate(item['steps']):
                hard = item['prm_hard_label'][i] if i < len(item['prm_hard_label']) else 'N/A'
                soft = item['prm_soft_label'][i] if i < len(item['prm_soft_label']) else 'N/A'
                soft_str = f"{soft:.3f}" if isinstance(soft, float) else str(soft)

                f.write(f"[Step {i+1}] PRM: hard={hard}, soft={soft_str}\n")
                f.write(step + "\n\n")

            f.write("-" * 80 + "\n")
            f.write("CLINICAL REVIEW:\n")
            f.write("-" * 80 + "\n")
            f.write("1. 각 Step이 임상적으로 정확한가? (1-5): \n")
            f.write("2. Step 분리가 적절한가? (1-5): \n")
            f.write("3. PRM reward가 임상 판단과 일치하는가? (1-5): \n")
            f.write("4. 추가 코멘트:\n\n")

    print(f"Saved detailed samples: {details_dir}")

    return json_path, excel_path


def print_sample_example(samples):
    """Print one example for quick verification"""
    if not samples:
        return

    print("\n" + "=" * 80)
    print("EXAMPLE SAMPLE")
    print("=" * 80)

    item = samples[0]
    print(f"\nQuestion ({item.get('data_source', 'unknown')}):")
    print(item.get('question', '')[:300] + "...")

    print(f"\nOptions:")
    for i, opt in enumerate(item.get('options', [])):
        print(f"  {chr(65+i)}. {opt[:50]}...")

    print(f"\nCorrect Answer: {item.get('correct_answer', '')}")

    solutions = item.get('solutions', [])
    if solutions:
        sol = solutions[0]
        print(f"\nModel Answer: {sol.get('answer', '')}")
        print(f"ORM Label: {sol.get('orm_label', '')}")
        print(f"PRM Hard Labels: {sol.get('prm_hard_label', [])}")
        print(f"PRM Soft Labels: {[f'{x:.2f}' for x in sol.get('prm_soft_label', [])]}")

        # Show first step
        prm_solution = sol.get('prm_processed_solution', '')
        if prm_solution:
            import re
            first_step = re.search(r'Step 1:.*?(?=Step 2:|$)', prm_solution, re.DOTALL)
            if first_step:
                print(f"\nFirst Step Preview:")
                print(first_step.group()[:300] + "...")


def main():
    print("=" * 80)
    print("Med-PRM Dataset Sampling for Clinical Review")
    print("Phase 1: 데이터 검토")
    print("=" * 80)

    # Load dataset
    dataset = load_medprm_dataset()

    # Analyze dataset
    analyze_dataset(dataset)

    # Sample dataset
    print(f"\n{'=' * 50}")
    print(f"Sampling {SAMPLE_SIZE} samples for clinical review")
    print("=" * 50)

    samples = sample_dataset(dataset, SAMPLE_SIZE, stratified=True)

    # Print example
    print_sample_example(samples)

    # Format for review
    review_data = format_for_review(samples)

    # Save
    json_path, excel_path = save_for_review(review_data, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Excel 파일을 임상쌤들께 전달:
   {excel_path}

2. 각 샘플에 대해 검토 요청:
   - Step Accuracy (1-5): 각 step이 임상적으로 정확한가?
   - Step Separation (1-5): step 분리가 적절한가?
   - Reward Alignment (1-5): PRM reward가 임상 판단과 일치하는가?
   - Comments: 추가 코멘트

3. 상세 샘플 검토 (detailed_samples/ 폴더):
   - 처음 20개 샘플의 상세 step-by-step 분석 파일

4. 검토 완료 후:
   - Phase 2 (BoN Implementation)로 진행
""")


if __name__ == "__main__":
    main()

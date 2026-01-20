#!/usr/bin/env python
# coding: utf-8
"""
Run PRM evaluation WITHOUT RAG (memory efficient).
- related_docs 무시
- max_token_len 4096에서 안정적으로 실행 가능
"""

import argparse
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRM evaluation WITHOUT RAG (memory efficient)."
    )
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to the saved model directory")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (e.g. '0')")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face access token (optional)")
    parser.add_argument("--input_json_file", type=str, required=True,
                        help="Path to input JSON file for evaluation")
    parser.add_argument("--output_json_file", type=str, required=True,
                        help="Path to save evaluation results")
    parser.add_argument("--process_solution_num", type=int, default=None,
                        help="Process only the first N solutions per question")
    parser.add_argument("--include_options", type=str,
                        choices=["yes", "no"], default="yes",
                        help="Include the options in the question text")
    parser.add_argument("--max_token_len", type=int, default=4096,
                        help="Max token length (RAG disabled)")
    parser.add_argument("--data_source_list", type=str, default=None,
                        help='JSON-array 형식 data_source 필터')
    return parser.parse_args()


def format_question_with_options(item):
    """질문 본문 (A) 옵션1 (B) 옵션2... 형태로 구성"""
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    return q + "".join(f" ({chr(ord('A') + i)}) {opt}"
                       for i, opt in enumerate(opts))


def main():
    args = parse_args()
    raw_src_arg = args.data_source_list

    print("====== 평가 설정 (RAG 비활성화) ======")
    print(f"모델 경로: {args.model_save_path}")
    print(f"입력 파일: {args.input_json_file}")
    print(f"출력 파일: {args.output_json_file}")
    print(f"Device: {args.device}")
    print(f"Max Token Length: {args.max_token_len}")
    print(f"⭐ RAG: 비활성화 (메모리 효율적)")
    print("=" * 50)

    if not raw_src_arg:
        filter_sources = []
    else:
        try:
            filter_sources = json.loads(raw_src_arg)
            assert isinstance(filter_sources, list)
        except Exception:
            raise ValueError("--data_source_list는 JSON 배열 형식이어야 합니다")

    if args.hf_token:
        login(args.hf_token)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("🔄 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_save_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    print(f"✅ 모델 로드 완료: {type(model).__name__}")

    # '+', '-' 토큰 ID
    plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
    minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]
    print(f"plus_id  : {plus_id} ({tokenizer.convert_ids_to_tokens([plus_id])})")
    print(f"minus_id : {minus_id} ({tokenizer.convert_ids_to_tokens([minus_id])})")

    # PRM 점수 계산 함수
    def get_prob(text, special_char=" ки"):
        """PRM 점수 계산"""
        encoded = tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        offsets = encoded["offset_mapping"][0]

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits[0]

        positions = [i for i, (s, e) in enumerate(offsets)
                     if text[s:e] == special_char]

        plus_probs, min_plus, final_plus = [], None, None
        for pos in positions:
            if pos >= logits.size(0):
                continue
            two = torch.stack([logits[pos][plus_id], logits[pos][minus_id]])
            probs = torch.softmax(two, dim=0)
            plus_probs.append(probs[0])

        if plus_probs:
            min_plus = torch.min(torch.stack(plus_probs)).item()
            final_plus = plus_probs[-1].item()

        return {
            "plus_probs": plus_probs,
            "min_plus_prob": min_plus,
            "final_plus_prob": final_plus
        }

    # JSON 처리 함수
    def process_json_with_prm():
        print("📂 JSON 파일 로드 중...")
        with open(args.input_json_file, encoding="utf-8") as f:
            data = json.load(f)

        if filter_sources:
            data = [d for d in data if d.get("data_source") in filter_sources]
        total = len(data)
        print(f"📋 처리할 데이터 항목 수: {total}")

        # 시스템 프롬프트 (RAG 없음)
        PRM_SYSTEM_PROMPT = (
            "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
            "In order to support the evaluation, the question and the explanation are provided. "
            "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
        )

        prm_correct = 0
        mv_correct = 0
        skip_count = 0

        with tqdm(total=total, desc="Processing Questions", unit="q") as pbar:
            for idx, item in enumerate(data):
                # 질문 문자열
                q_text = (format_question_with_options(item)
                          if args.include_options == "yes"
                          else item.get("question", ""))

                # 솔루션 수 제한
                if args.process_solution_num is not None:
                    item["solutions"] = item["solutions"][:args.process_solution_num]
                sols = item["solutions"]

                # 솔루션마다 PRM 점수 부여
                for sol_idx, sol in enumerate(sols):
                    sol_text = sol.get("prm_processed_solution", "")
                    # RAG 없음: Question + Explanation 만
                    user_content = f"Question: {q_text}\n\nExplanation: {sol_text}"

                    messages = [
                        {"role": "system", "content": PRM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                    raw = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    res = get_prob(raw, special_char=" ки")
                    plus_probs = [p.item() for p in res["plus_probs"]]
                    sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
                    sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
                    sol["PRM_score_list"] = plus_probs

                    if res["min_plus_prob"] == float("-inf"):
                        skip_count += 1

                # PRM 기반 정답 여부
                valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
                prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
                if prm_pred and prm_pred.get("score", 0) == 1:
                    prm_correct += 1

                # Majority voting 기반 정답 여부
                if sols:
                    most_common_ans, _ = Counter(s["answer"] for s in sols).most_common(1)[0]
                    mv_sols = [s for s in sols if s["answer"] == most_common_ans]
                    if any(s.get("score", 0) == 1 for s in mv_sols):
                        mv_correct += 1

                current_prm_acc = (prm_correct / (idx + 1)) * 100
                current_mv_acc = (mv_correct / (idx + 1)) * 100

                pbar.set_description(f"Q{idx+1}/{total}")
                pbar.set_postfix(
                    PRM=f"{prm_correct}/{idx+1} ({current_prm_acc:.1f}%)",
                    MV=f"{mv_correct}/{idx+1} ({current_mv_acc:.1f}%)",
                    Skip=skip_count
                )
                pbar.update(1)

        print("\n💾 결과 저장 중...")
        with open(args.output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Done. Results saved to {args.output_json_file}")
        print(f"PRM Accuracy : {prm_correct}/{total} ({100*prm_correct/total:.2f}%)")
        print(f"Maj-Vote Acc : {mv_correct}/{total} ({100*mv_correct/total:.2f}%)")
        print(f"\n📊 Statistics:")
        print(f"   - Total solutions: {total * 64}")
        print(f"   - Skipped: {skip_count}")
        print(f"   - Skip ratio: {100*skip_count/(total*64):.2f}%")

    process_json_with_prm()


if __name__ == "__main__":
    main()

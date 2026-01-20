#!/usr/bin/env python
# coding: utf-8
"""
Run PRM evaluation with optional RAG support.
WITH CHECKPOINT SAVING - saves every 100 questions
"""

import argparse
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import accelerate
from collections import Counter

# ----------------------------------------------------------------------
# 1. 인자 파서
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRM evaluation (RAG on/off selectable)."
    )
    parser.add_argument("--model_save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--input_json_file", type=str, required=True)
    parser.add_argument("--output_json_file", type=str, required=True)
    parser.add_argument("--process_solution_num", type=int, default=None)
    parser.add_argument("--include_options", type=str, choices=["yes", "no"], default="yes")
    parser.add_argument("--use_rag", type=str, choices=["yes", "no"], default="yes")
    parser.add_argument("--max_token_len", type=int, default=4096)
    parser.add_argument("--use_orm", choices=["yes", "no"], default="no")
    parser.add_argument("--data_source_list", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N questions")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint file")
    return parser.parse_args()


# ----------------------------------------------------------------------
# 2. 유틸 함수
# ----------------------------------------------------------------------
def format_question_with_options(item):
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    return q + "".join(f" ({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(opts))


def truncate_related_docs(docs, tokenizer, max_total_len: int, reserve_for_q_and_sol: int = 1024):
    kept, used = [], 0
    budget = max_total_len - reserve_for_q_and_sol
    for doc in docs:
        tok_len = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        if used + tok_len + 1 > budget:
            break
        kept.append(doc)
        used += tok_len + 1
    return kept


# ----------------------------------------------------------------------
# 3. 메인 로직
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    raw_src_arg = args.data_source_list

    print("====== 평가 설정 ======")
    print(f"모델 경로: {args.model_save_path}")
    print(f"입력 파일: {args.input_json_file}")
    print(f"출력 파일: {args.output_json_file}")
    print(f"RAG 사용: {args.use_rag}")
    print(f"체크포인트 간격: {args.checkpoint_interval}")
    print("=====================")

    if not raw_src_arg:
        filter_sources = []
    else:
        try:
            filter_sources = json.loads(raw_src_arg)
            assert isinstance(filter_sources, list)
        except Exception:
            raise ValueError("--data_source_list 는 JSON 배열 형식이어야 합니다")

    if args.hf_token:
        login(args.hf_token)

    target_device = args.device if args.device else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = target_device
    print(f"🎯 CUDA_VISIBLE_DEVICES set to: {target_device}")

    print("🔄 모델 로드 중...")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    attn_impl = "eager" if args.no_flash_attn else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_save_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    print(f"✅ 모델 로드 완료")

    plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
    minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]

    # --------------------------------------------------------------
    # PRM 점수 계산
    # --------------------------------------------------------------
    def get_prob(text, special_char=" ки"):
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        if input_ids.size(1) > 4096:
            print(f"Skip: {input_ids.size(1)} tokens")
            return None
        offsets = encoded["offset_mapping"][0]

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits[0]

        positions = [i for i, (s, e) in enumerate(offsets) if text[s:e] == special_char]

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

        return {"plus_probs": plus_probs, "min_plus_prob": min_plus, "final_plus_prob": final_plus}

    # --------------------------------------------------------------
    # JSON 처리
    # --------------------------------------------------------------
    print("📂 JSON 파일 로드 중...")
    with open(args.input_json_file, encoding="utf-8") as f:
        data = json.load(f)

    if filter_sources:
        data = [d for d in data if d.get("data_source") in filter_sources]
    total = len(data)
    print(f"📋 처리할 데이터 항목 수: {total}")

    # Resume from checkpoint
    start_idx = 0
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, encoding="utf-8") as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get("last_processed_idx", 0) + 1
        data[:start_idx] = checkpoint.get("processed_data", data[:start_idx])
        print(f"📌 체크포인트에서 재개: {start_idx}부터 시작")

    RAG_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
    )
    PRM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
    )
    ORM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the overall quality and correctness of the final answer in the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the final answer is incorrect or not well-supported, output -. If the final answer is correct and well-supported, output +."
    )

    prm_correct = 0
    mv_correct = 0
    processed_count = start_idx

    # Count previous correct answers
    for item in data[:start_idx]:
        sols = item.get("solutions", [])
        valid = [s for s in sols if s.get("PRM_min_score", float("-inf")) != float("-inf")]
        if valid:
            prm_pred = max(valid, key=lambda s: s.get("PRM_min_score", float("-inf")))
            if prm_pred.get("score", 0) == 1:
                prm_correct += 1
        answers = [s.get("answer", "?") for s in sols]
        if answers:
            mv_answer = Counter(answers).most_common(1)[0][0]
            if mv_answer == item.get("correct_answer"):
                mv_correct += 1

    checkpoint_file = args.output_json_file.replace(".json", "_checkpoint.json")

    with tqdm(total=total, initial=start_idx, desc="Processing Questions", unit="q") as pbar:
        for idx in range(start_idx, total):
            item = data[idx]
            q_text = (format_question_with_options(item) if args.include_options == "yes"
                      else item.get("question", ""))

            if args.process_solution_num is not None:
                item["solutions"] = item["solutions"][:args.process_solution_num]
            sols = item["solutions"]

            if args.use_rag == "yes":
                docs = truncate_related_docs(item.get("related_docs", []), tokenizer,
                                              max_total_len=args.max_token_len, reserve_for_q_and_sol=1024)
                doc_block = "".join(f"Document {i+1}: {d}\n\n" for i, d in enumerate(docs))
                system_prompt = RAG_SYSTEM_PROMPT
                sol_key = "prm_processed_solution"
            else:
                doc_block = ""
                if args.use_orm == "yes":
                    system_prompt = ORM_SYSTEM_PROMPT
                    sol_key = "orm_processed_solution"
                else:
                    system_prompt = PRM_SYSTEM_PROMPT
                    sol_key = "prm_processed_solution"

            for sol in sols:
                sol_text = sol.get(sol_key, "")
                user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {sol_text}"
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}]
                raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                res = get_prob(raw, special_char=" ки")
                if res is None:
                    sol["PRM_min_score"] = float("-inf")
                    sol["PRM_score"] = float("-inf")
                    sol["PRM_score_list"] = []
                    continue
                plus_probs = [p.item() for p in res["plus_probs"]]
                sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
                sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
                sol["PRM_score_list"] = plus_probs

            # PRM 정답 여부
            valid = [s for s in sols if s["PRM_min_score"] != float("-inf")]
            prm_pred = max(valid, key=lambda s: s["PRM_min_score"]) if valid else None
            if prm_pred and prm_pred.get("score", 0) == 1:
                prm_correct += 1

            # MV 정답 여부
            answers = [s.get("answer", "?") for s in sols]
            mv_answer = Counter(answers).most_common(1)[0][0] if answers else "?"
            if mv_answer == item.get("correct_answer"):
                mv_correct += 1

            processed_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"MV={mv_correct}/{processed_count} ({100*mv_correct/processed_count:.1f}%), "
                                 f"PRM={prm_correct}/{processed_count} ({100*prm_correct/processed_count:.1f}%)")

            # 체크포인트 저장
            if processed_count % args.checkpoint_interval == 0:
                checkpoint_data = {
                    "last_processed_idx": idx,
                    "processed_data": data[:idx+1],
                    "prm_correct": prm_correct,
                    "mv_correct": mv_correct
                }
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False)
                print(f"\n💾 체크포인트 저장: {checkpoint_file} (Q{idx+1}/{total})")

    # 최종 결과 저장
    print(f"\n📊 최종 결과: MV={mv_correct}/{total} ({100*mv_correct/total:.1f}%), "
          f"PRM={prm_correct}/{total} ({100*prm_correct/total:.1f}%)")

    with open(args.output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 결과 저장 완료: {args.output_json_file}")


if __name__ == "__main__":
    main()

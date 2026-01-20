#!/usr/bin/env python
# coding: utf-8
"""
Run PRM evaluation with RAG caching optimization.
- 같은 related_docs는 캐싱 → 메모리 절감
- max_token_len 4096 안정화
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
import hashlib


# ===== CACHING 추가 =====
class RAGDocumentCache:
    """RAG 문서 캐싱 클래스"""
    def __init__(self):
        self.doc_cache = {}      # 캐시: doc_hash -> truncated_docs
        self.doc_block_cache = {} # 캐시: doc_hash -> doc_block string

    def get_cache_key(self, docs_list):
        """관련 문서 리스트의 해시 키 생성"""
        doc_str = json.dumps(docs_list, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(doc_str.encode()).hexdigest()

    def get_truncated_docs(self, docs, tokenizer, max_total_len, reserve_for_q_and_sol):
        """캐싱된 truncate_related_docs"""
        cache_key = self.get_cache_key(docs)

        if cache_key in self.doc_cache:
            return self.doc_cache[cache_key]

        # 캐시 미스: 실행 후 저장
        truncated = truncate_related_docs(docs, tokenizer, max_total_len, reserve_for_q_and_sol)
        self.doc_cache[cache_key] = truncated
        return truncated

    def get_doc_block(self, docs, tokenizer, max_total_len, reserve_for_q_and_sol):
        """캐싱된 doc_block 생성"""
        cache_key = self.get_cache_key(docs)

        if cache_key in self.doc_block_cache:
            return self.doc_block_cache[cache_key]

        # 캐시 미스: 생성 후 저장
        truncated = self.get_truncated_docs(docs, tokenizer, max_total_len, reserve_for_q_and_sol)
        doc_block = "".join(f"Document {i+1}: {d}\n\n" for i, d in enumerate(truncated))
        self.doc_block_cache[cache_key] = doc_block
        return doc_block


# ===== 기존 함수들 =====
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRM evaluation with RAG caching (optimized)."
    )

    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to the saved model directory")
    parser.add_argument("--device", type=str, default="",
                        help="CUDA visible devices (e.g. '0')")
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

    parser.add_argument("--use_rag", type=str,
                        choices=["yes", "no"], default="yes",
                        help="'yes': use related_docs / 'no': base PRM only")
    parser.add_argument("--max_token_len", type=int, default=4096,
                        help="Token budget when use_rag is 'yes'")
    parser.add_argument("--use_orm", choices=["yes", "no"], default="no",
                   help="'yes': use orm_processed_solution when RAG is off")
    parser.add_argument(
        "--data_source_list", type=str, default=None,
        help='JSON-array 형식으로 추론할 data_source 이름들만 지정 '
            '(예: \'["medqa","pubmedqa"]\'). 빈 리스트면 전체 사용'
    )

    return parser.parse_args()


def format_question_with_options(item):
    """질문 본문 (A) 옵션1 (B) 옵션2... 형태로 구성"""
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    return q + "".join(f" ({chr(ord('A') + i)}) {opt}"
                       for i, opt in enumerate(opts))


def truncate_related_docs(docs, tokenizer,
                          max_total_len: int,
                          reserve_for_q_and_sol: int = 1024):
    """관련 문서를 토큰 수 한도 내로 자르는 함수 (RAG 모드 전용)"""
    kept, used = [], 0
    budget = max_total_len - reserve_for_q_and_sol
    for doc in docs:
        tok_len = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        if used + tok_len + 1 > budget:
            break
        kept.append(doc)
        used += tok_len + 1
    return kept


def main():
    args = parse_args()
    raw_src_arg = args.data_source_list

    print("====== 평가 설정 (RAG 캐싱 최적화) ======")
    print(f"모델 경로: {args.model_save_path}")
    print(f"입력 파일: {args.input_json_file}")
    print(f"출력 파일: {args.output_json_file}")
    print(f"RAG 사용: {args.use_rag}")
    print(f"ORM 사용: {args.use_orm}")
    print(f"Max Token Length: {args.max_token_len}")
    print(f"⭐ RAG 캐싱: 활성화됨")
    print("=" * 50)

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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("🔄 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_save_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    print(f"✅ 모델 로드 완료: {type(model).__name__}")

    # '+', '-' 토큰 ID
    plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
    minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]
    print(f"plus_id  : {plus_id} ({tokenizer.convert_ids_to_tokens([plus_id])})")
    print(f"minus_id : {minus_id} ({tokenizer.convert_ids_to_tokens([minus_id])})")

    # ========== RAG 캐시 초기화 ==========
    rag_cache = RAGDocumentCache()

    # PRM 점수 계산
    def get_prob(text, special_char=" ки"):
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

    # JSON 처리
    def process_json_with_prm():
        print("📂 JSON 파일 로드 중...")
        with open(args.input_json_file, encoding="utf-8") as f:
            data = json.load(f)

        if filter_sources:
            data = [d for d in data if d.get("data_source") in filter_sources]
        total = len(data)
        print(f"📋 처리할 데이터 항목 수: {total}")

        RAG_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step. "
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
        mv_correct  = 0

        with tqdm(total=total, desc="Processing Questions", unit="q") as pbar:
            for idx, item in enumerate(data):
                q_text = (format_question_with_options(item)
                          if args.include_options == "yes"
                          else item.get("question", ""))

                if args.process_solution_num is not None:
                    item["solutions"] = item["solutions"][:args.process_solution_num]
                sols = item["solutions"]

                # ========== RAG 캐싱 적용 ==========
                if args.use_rag == "yes":
                    related_docs = item.get("related_docs", [])
                    # 캐싱된 doc_block 획득
                    doc_block = rag_cache.get_doc_block(
                        related_docs,
                        tokenizer,
                        max_total_len=args.max_token_len,
                        reserve_for_q_and_sol=1024
                    )
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

                # 솔루션마다 PRM 점수 부여
                for sol_idx, sol in enumerate(sols):
                    sol_text = sol.get(sol_key, "")
                    user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {sol_text}"

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content}
                    ]
                    raw = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    res = get_prob(raw, special_char=" ки")
                    plus_probs = [p.item() for p in res["plus_probs"]]
                    sol["PRM_min_score"] = res["min_plus_prob"] if res["min_plus_prob"] is not None else float("-inf")
                    sol["PRM_score"] = res["final_plus_prob"] if res["final_plus_prob"] is not None else float("-inf")
                    sol["PRM_score_list"] = plus_probs

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
                    CacheSize=len(rag_cache.doc_block_cache)
                )
                pbar.update(1)

        print("\n💾 결과 저장 중...")
        with open(args.output_json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Done. Results saved to {args.output_json_file}")
        print(f"PRM Accuracy : {prm_correct}/{total} ({100*prm_correct/total:.2f}%)")
        print(f"Maj-Vote Acc : {mv_correct}/{total} ({100*mv_correct/total:.2f}%)")
        print(f"\n📊 RAG Cache Statistics:")
        print(f"   - Doc blocks cached: {len(rag_cache.doc_block_cache)}")
        print(f"   - Cache hit rate: {(len(rag_cache.doc_block_cache) / max(1, total)) * 100:.1f}%")

    process_json_with_prm()


if __name__ == "__main__":
    main()

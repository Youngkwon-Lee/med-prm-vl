import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('./model')
model = AutoModelForCausalLM.from_pretrained(
    './model',
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)
print("Model loaded!")

plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]

RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
    "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
    "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
)

def get_min_plus(item, sol):
    docs = item.get('related_docs', [])[:3]
    doc_block = ''.join(f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs))
    q_text = item['question']
    pps = sol.get('prm_processed_solution', '')

    user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {pps}"
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    encoded = tokenizer(raw, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = encoded["input_ids"].to(model.device)

    if input_ids.size(1) > 4096:
        return float('-inf')

    attention_mask = encoded["attention_mask"].to(model.device)
    offsets = encoded["offset_mapping"][0]

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits[0]

    positions = [i for i, (s, e) in enumerate(offsets) if s < len(raw) and raw[s:e] == " ки"]

    plus_probs = []
    for pos in positions:
        if pos >= logits.size(0):
            continue
        two = torch.stack([logits[pos][plus_id], logits[pos][minus_id]])
        probs = torch.softmax(two, dim=0)
        plus_probs.append(probs[0].item())

    return min(plus_probs) if plus_probs else float('-inf')

# 테스트: 처음 5개 질문 (정답 솔루션이 있는 것만)
d = json.load(open('input.json'))

# Q1, Q2 테스트 (정답 솔루션이 많은 케이스)
test_questions = [1, 2, 5]  # Q1: 40개 정답, Q2: 64개 정답, Q5: 8개 정답

for q_idx in test_questions:
    item = d[q_idx]
    correct = item['correct_answer']
    sols = item['solutions']

    print(f"\n{'='*60}")
    print(f"Q{q_idx}: Correct={correct}, #Correct sols={len([s for s in sols if s.get('score')==1])}")

    # 처음 16개 솔루션만 테스트 (시간 절약)
    scores = []
    for i, sol in enumerate(sols[:16]):
        min_plus = get_min_plus(item, sol)
        answer = sol.get('answer', '?')
        score = sol.get('score', 0)
        scores.append((i, answer, score, min_plus))
        print(f"  Sol{i}: ans={answer}, correct={score}, min_p={min_plus:.4f}")

    # PRM 선택
    best = max(scores, key=lambda x: x[3])
    print(f"\n  PRM selects: Sol{best[0]} (ans={best[1]}, correct={best[2]})")
    print(f"  {'✅ PRM CORRECT!' if best[2]==1 else '❌ PRM WRONG!'}")

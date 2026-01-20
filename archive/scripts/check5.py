import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# 데이터 로드
d = json.load(open('input.json'))[:1]
item = d[0]

# RAG 프롬프트
docs = item.get('related_docs', [])[:3]
doc_block = ''.join(f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs))
RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
    "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
    "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
)
q_text = item['question']

def get_min_plus(sol):
    pps = sol.get('prm_processed_solution', '')
    user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {pps}"
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    encoded = tokenizer(raw, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    offsets = encoded["offset_mapping"][0]

    if input_ids.size(1) > 4096:
        return None

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

    return min(plus_probs) if plus_probs else None

print(f"\nQuestion: {item['question'][:100]}...")
print(f"Correct answer: {item['correct_answer']}")
print(f"\n{'='*60}")
print(f"{'Sol#':<5} {'Answer':<8} {'Score':<6} {'Min P(+)':<10} {'Status'}")
print(f"{'='*60}")

# 처음 10개 솔루션만 테스트
correct_sols = []
wrong_sols = []

for i, sol in enumerate(item['solutions'][:10]):
    min_plus = get_min_plus(sol)
    answer = sol.get('answer', '?')
    score = sol.get('score', 0)
    status = "CORRECT" if score == 1 else "WRONG"

    if min_plus:
        print(f"{i:<5} {answer:<8} {score:<6} {min_plus:.4f}     {status}")
        if score == 1:
            correct_sols.append((i, min_plus))
        else:
            wrong_sols.append((i, min_plus))

print(f"\n{'='*60}")
if correct_sols:
    best_correct = max(correct_sols, key=lambda x: x[1])
    print(f"Best CORRECT solution: Sol#{best_correct[0]} with Min P(+)={best_correct[1]:.4f}")
if wrong_sols:
    best_wrong = max(wrong_sols, key=lambda x: x[1])
    print(f"Best WRONG solution: Sol#{best_wrong[0]} with Min P(+)={best_wrong[1]:.4f}")

if correct_sols and wrong_sols:
    if best_correct[1] > best_wrong[1]:
        print("\n✅ PRM would select CORRECT answer!")
    else:
        print("\n❌ PRM would select WRONG answer!")

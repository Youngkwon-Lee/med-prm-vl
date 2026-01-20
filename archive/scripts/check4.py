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

# 토큰 IDs
plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]

# 실제 데이터 로드
d = json.load(open('input.json'))[:1]
item = d[0]
sol = item['solutions'][0]

# RAG 프롬프트 구성
docs = item.get('related_docs', [])[:3]
doc_block = ''.join(f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs))

RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
    "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
    "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
)

q_text = item['question']
pps = sol.get('prm_processed_solution', '')

user_content = f"{doc_block}Question: {q_text}\n\nExplanation: {pps}"

messages = [
    {"role": "system", "content": RAG_SYSTEM_PROMPT},
    {"role": "user", "content": user_content}
]

raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 토큰화
encoded = tokenizer(raw, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
input_ids = encoded["input_ids"].to(model.device)
attention_mask = encoded["attention_mask"].to(model.device)
offsets = encoded["offset_mapping"][0]

print(f"\nTotal tokens: {input_ids.size(1)}")

# 모델 추론
with torch.no_grad():
    logits = model(input_ids, attention_mask=attention_mask).logits[0]

# ' ки' 위치 찾기
special_char = " ки"
positions = [i for i, (s, e) in enumerate(offsets) if s < len(raw) and raw[s:e] == special_char]

print(f"Found {len(positions)} ' ки' positions")

# 각 위치에서 +/- 확률 계산
print("\n=== Step-wise +/- Probabilities ===")
plus_probs = []
for idx, pos in enumerate(positions):
    if pos >= logits.size(0):
        print(f"Step {idx+1}: Position {pos} out of range")
        continue

    two = torch.stack([logits[pos][plus_id], logits[pos][minus_id]])
    probs = torch.softmax(two, dim=0)
    plus_prob = probs[0].item()
    minus_prob = probs[1].item()
    plus_probs.append(plus_prob)

    print(f"Step {idx+1}: P(+)={plus_prob:.4f}, P(-)={minus_prob:.4f}")

if plus_probs:
    print(f"\nMin P(+): {min(plus_probs):.4f}")
    print(f"Final P(+): {plus_probs[-1]:.4f}")
    print(f"Solution score (ground truth): {sol.get('score', 'N/A')}")
    print(f"Solution answer: {sol.get('answer', 'N/A')}")
    print(f"Correct answer: {item.get('correct_answer', 'N/A')}")

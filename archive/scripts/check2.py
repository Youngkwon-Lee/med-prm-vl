import json
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('./model')

# 데이터 로드
d = json.load(open('input.json'))[:10]  # 처음 10개 질문

total_sols = 0
over_3200 = 0
under_3200 = 0

for item in d:
    for sol in item['solutions'][:5]:  # 질문당 5개 솔루션만
        pps = sol.get('prm_processed_solution', '')

        # RAG 문서 포함 시뮬레이션
        docs = item.get('related_docs', [])[:3]
        doc_block = ''.join(f"Document {i+1}: {doc}\n\n" for i, doc in enumerate(docs))

        full_text = f"{doc_block}Question: {item['question']}\n\nExplanation: {pps}"
        tokens = tokenizer(full_text, add_special_tokens=True)['input_ids']

        total_sols += 1
        if len(tokens) > 3200:
            over_3200 += 1
        else:
            under_3200 += 1

print(f"Total solutions checked: {total_sols}")
print(f"Over 3200 tokens (SKIPPED): {over_3200} ({100*over_3200/total_sols:.1f}%)")
print(f"Under 3200 tokens (OK): {under_3200} ({100*under_3200/total_sols:.1f}%)")

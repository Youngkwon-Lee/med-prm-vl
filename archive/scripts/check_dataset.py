import json
from collections import Counter

print("=== Dataset Info ===")
d = json.load(open('input.json'))

# data_source 확인
sources = [item.get('data_source', 'unknown') for item in d]
print(f'Total questions: {len(d)}')
print(f'Data sources: {Counter(sources)}')

# 샘플 질문 확인
print("\n=== Sample Questions ===")
for i in range(min(5, len(d))):
    print(f"Q{i}: {d[i].get('data_source', 'N/A')} - {d[i]['question'][:80]}...")

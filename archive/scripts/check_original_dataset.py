import json
from collections import Counter

print("=== Original Med-PRM Test Dataset ===")

d1 = json.load(open('/home2/gun3856/med-prm-vl/dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json'))
d2 = json.load(open('/home2/gun3856/med-prm-vl/dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset_part2.json'))

print(f'Part 1: {len(d1)} questions')
print(f'Part 2: {len(d2)} questions')
print(f'Total: {len(d1) + len(d2)} questions')

# 합치기
all_data = d1 + d2

# 데이터 소스 확인
sources = [item.get('data_source', 'unknown') for item in all_data]
print(f'\nData sources: {Counter(sources)}')

# 샘플 확인
print(f'\nSample keys: {list(all_data[0].keys())}')
print(f'Sample data_source: {all_data[0].get("data_source")}')

# 벤치마크별 통계
print(f'\n=== Benchmark Distribution ===')
for source, count in sorted(Counter(sources).items(), key=lambda x: x[1], reverse=True):
    print(f'{source}: {count}')

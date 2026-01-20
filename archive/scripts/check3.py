import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model')

# ' ки' 토큰화 확인
test_text = "Step 1: This is a test. ки Step 2: Another test. ки"
encoded = tokenizer(test_text, return_offsets_mapping=True, add_special_tokens=True)

print("=== ' ки' 토큰화 테스트 ===")
print(f"원본 텍스트: {test_text}")
print(f"토큰 IDs: {encoded['input_ids']}")
print(f"토큰들: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")

# offset_mapping으로 ' ки' 위치 찾기
offsets = encoded['offset_mapping']
special_char = " ки"
positions = []
for i, (s, e) in enumerate(offsets):
    if s < len(test_text) and test_text[s:e] == special_char:
        positions.append(i)
        print(f"Found '{special_char}' at position {i}, offset ({s}, {e})")

print(f"\n' ки' 발견 위치: {positions}")
print(f"발견 개수: {len(positions)} (예상: 2)")

# + / - 토큰 ID 확인
plus_id = tokenizer(" +", add_special_tokens=False)["input_ids"][0]
minus_id = tokenizer(" -", add_special_tokens=False)["input_ids"][0]
print(f"\nplus_id: {plus_id} -> {tokenizer.convert_ids_to_tokens([plus_id])}")
print(f"minus_id: {minus_id} -> {tokenizer.convert_ids_to_tokens([minus_id])}")

# 실제 데이터로 테스트
print("\n=== 실제 데이터 테스트 ===")
d = json.load(open('input.json'))[:1]
sol = d[0]['solutions'][0]
pps = sol.get('prm_processed_solution', '')

encoded2 = tokenizer(pps, return_offsets_mapping=True, add_special_tokens=True)
offsets2 = encoded2['offset_mapping']

positions2 = []
for i, (s, e) in enumerate(offsets2):
    if s < len(pps) and pps[s:e] == special_char:
        positions2.append(i)

print(f"실제 솔루션에서 ' ки' 발견: {len(positions2)}개")
print(f"pps.count(' ки'): {pps.count(' ки')}")

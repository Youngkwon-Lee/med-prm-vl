#!/usr/bin/env python3
"""
Med-PRM 스크립트 패치 v4 - 중간 저장 기능 추가
- 50문제마다 자동 저장
- 중단해도 결과 보존
"""
import os

SCRIPT_PATH = os.path.expanduser("~/med-prm-vl/python/4_scoring_PRM.py")
BACKUP_PATH = os.path.expanduser("~/med-prm-vl/python/4_scoring_PRM.py.backup")

with open(BACKUP_PATH, 'r') as f:
    content = f.read()

# 1. 8bit 양자화
content = content.replace(
    'device_map="auto"',
    'device_map="auto",\n        load_in_8bit=True'
)

# 2. 토큰 제한 2500
old_offsets = '        offsets = encoded["offset_mapping"][0]'
new_offsets = '''        if input_ids.size(1) > 2500:
            print(f"Skip: {input_ids.size(1)} tokens")
            return None
        offsets = encoded["offset_mapping"][0]'''
content = content.replace(old_offsets, new_offsets)

# 3. None 처리
old_res = '                    res = get_prob(raw, special_char=" ки")'
new_res = '''                    res = get_prob(raw, special_char=" ки")
                    if res is None:
                        sol["PRM_min_score"] = float("-inf")
                        sol["PRM_final_score"] = float("-inf")
                        sol["PRM_score_list"] = []
                        continue'''
content = content.replace(old_res, new_res)

# 4. 중간 저장 추가 (tqdm loop 내부)
old_pbar = '''    with tqdm(total=len(data), desc="Processing") as pbar:
        for idx, question_data in enumerate(data):'''

new_pbar = '''    checkpoint_interval = 50
    with tqdm(total=len(data), desc="Processing") as pbar:
        for idx, question_data in enumerate(data):'''

content = content.replace(old_pbar, new_pbar)

# 5. 중간 저장 로직 (loop 내부 끝에 추가)
old_pbar_update = '''            pbar.update(1)'''
new_pbar_update = '''            pbar.update(1)

            # 중간 저장 (50문제마다)
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_file = args.output_json_file.replace('.json', f'_checkpoint_{idx+1}.json')
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(data[:idx+1], f, ensure_ascii=False, indent=2)
                    print(f"\\n💾 Checkpoint saved: {checkpoint_file}")
                except Exception as e:
                    print(f"\\n⚠️ Checkpoint save failed: {e}")'''

content = content.replace(old_pbar_update, new_pbar_update)

with open(SCRIPT_PATH, 'w') as f:
    f.write(content)

print("Done! Patched v4 (checkpoint every 50 questions):", SCRIPT_PATH)

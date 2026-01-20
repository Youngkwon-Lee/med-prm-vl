#!/usr/bin/env python3
"""
Med-PRM 스크립트 패치 v2 - None 처리 개선
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

# 2. 긴 시퀀스 스킵
old_offsets = '        offsets = encoded["offset_mapping"][0]'
new_offsets = '''        if input_ids.size(1) > 1500:
            print(f"Skip: {input_ids.size(1)} tokens")
            return None
        offsets = encoded["offset_mapping"][0]'''
content = content.replace(old_offsets, new_offsets)

# 3. None 처리 개선 - continue 대신 기본값 설정
old_res = '                    res = get_prob(raw, special_char=" ки")'
new_res = '''                    res = get_prob(raw, special_char=" ки")
                    if res is None:
                        sol["PRM_min_score"] = float("-inf")
                        sol["PRM_final_score"] = float("-inf")
                        sol["PRM_score_list"] = []
                        continue'''
content = content.replace(old_res, new_res)

with open(SCRIPT_PATH, 'w') as f:
    f.write(content)

print("Done! Patched v2:", SCRIPT_PATH)

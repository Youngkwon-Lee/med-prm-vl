#!/usr/bin/env python3
"""
Med-PRM 스크립트 패치 - OOM 방지용
사용법: python patch_prm.py
"""
import os

SCRIPT_PATH = os.path.expanduser("~/med-prm-vl/python/4_scoring_PRM.py")
BACKUP_PATH = os.path.expanduser("~/med-prm-vl/python/4_scoring_PRM.py.backup")

# 백업에서 읽기
with open(BACKUP_PATH, 'r') as f:
    content = f.read()

# 1. 8bit 양자화 추가
content = content.replace(
    'device_map="auto"',
    'device_map="auto",\n        load_in_8bit=True'
)

# 2. 긴 시퀀스 스킵 (get_prob 함수 내부, offsets 라인 앞에)
old_offsets = '        offsets = encoded["offset_mapping"][0]'
new_offsets = '''        # OOM 방지: 긴 시퀀스 스킵
        if input_ids.size(1) > 1500:
            print(f"Skip: {input_ids.size(1)} tokens")
            return None
        offsets = encoded["offset_mapping"][0]'''
content = content.replace(old_offsets, new_offsets)

# 3. None 처리 추가 (get_prob 호출 후)
old_res = '                    res = get_prob(raw, special_char=" ки")'
new_res = '''                    res = get_prob(raw, special_char=" ки")
                    if res is None:
                        continue'''
content = content.replace(old_res, new_res)

# 저장
with open(SCRIPT_PATH, 'w') as f:
    f.write(content)

print("Done! Patched:", SCRIPT_PATH)

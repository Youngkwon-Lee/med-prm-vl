#!/bin/bash
# Step 3: HPC에서 실행
# 스킵된 항목 재처리

cd ~/med-prm-vl

echo "=================================================="
echo "Step 3: Retest Skipped Items"
echo "=================================================="

# 파일 존재 확인
if [ ! -f "input_skipped_items.json" ]; then
    echo "[ERROR] input_skipped_items.json not found!"
    echo "Run step1_verify_and_extract.py first"
    exit 1
fi

echo "[OK] Found input_skipped_items.json"

# 재처리 실행
echo ""
echo "Starting PRM retest for skipped items..."
echo "Model: ./model"
echo "Input: input_skipped_items.json"
echo "Output: output/medprm_scores_skipped_retested.json"
echo "Token limit: 5000"
echo ""

nohup python python/4_scoring_PRM.py \
  --model_save_path ./model \
  --input_json_file ./input_skipped_items.json \
  --output_json_file ./output/medprm_scores_skipped_retested.json \
  --device 0 \
  --use_rag yes \
  --max_token_len 5000 > log_retest_skipped.out 2>&1 &

PID=$!
echo "[OK] Process started: PID $PID"
echo "Log file: log_retest_skipped.out"
echo ""
echo "Monitor progress with: tail -f log_retest_skipped.out"

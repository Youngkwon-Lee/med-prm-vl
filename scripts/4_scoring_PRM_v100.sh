#!/bin/bash
# Med-PRM Scoring Script for V100 GPU (HPC Innovation Hub)
# VM: gun3856@10.246.246.111
# GPU: NVIDIA V100 16GB × 2

############################################
# Load Environment Variables
############################################
if [ -f ".env" ]; then
  source ".env"
fi

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set. Please check .env file."
  exit 1
fi

############################################
# V100 Optimized Settings
############################################
USE_RAG="yes"
USE_ORM="no"

# V100 메모리 최적화: 64 → 32
PROCESS_SOLUTION_NUM=32

# V100: float16 사용 (bfloat16 미지원)
DTYPE="float16"

MODEL_PATHS=(
"model_train/llama-3.1-medprm-reward-v1.0"
)
INPUT_JSON="dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json"

# V100 2장 사용 (0, 1)
GPUS=(0)

OUTPUT_DIR="dataset/dataset_4_scored_dataset"
MAX_TOKEN_LEN=4096
INCLUDE_OPTIONS="no"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

BASE_INPUT_NAME="$(basename "$INPUT_JSON" .json)"
FIRST_DATA_SOURCE=$(echo $DATA_SOURCE_LIST | sed -E 's/\[\"([^\"]+)\".*/\1/')

echo "============================================"
echo "Med-PRM Scoring - V100 Optimized"
echo "============================================"
echo "dtype: $DTYPE"
echo "process_solution_num: $PROCESS_SOLUTION_NUM"
echo "GPU: ${GPUS[@]}"
echo "============================================"

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    GPU="${GPUS[$i]}"
    MODEL_BASENAME="${MODEL_PATH##*/}"
    OUTPUT_JSON="${OUTPUT_DIR}/${MODEL_BASENAME}_v100_sol${PROCESS_SOLUTION_NUM}_${BASE_INPUT_NAME}.json"
    LOG_FILE="${LOG_DIR}/TEST_V100_$(date +'%Y%m%d_%H%M%S')_${MODEL_BASENAME}.log"

    echo "Starting evaluation: ${MODEL_BASENAME}" | tee -a "$LOG_FILE"
    echo "Output: $OUTPUT_JSON" | tee -a "$LOG_FILE"

    python python/4_scoring_PRM.py \
        --model_save_path "$MODEL_PATH" \
        --input_json_file "$INPUT_JSON" \
        --output_json_file "$OUTPUT_JSON" \
        --device "$GPU" \
        --hf_token "$HF_TOKEN" \
        --use_rag "$USE_RAG" \
        --max_token_len "$MAX_TOKEN_LEN" \
        --include_options "$INCLUDE_OPTIONS" \
        --use_orm "$USE_ORM" \
        --process_solution_num "$PROCESS_SOLUTION_NUM" \
        --dtype "$DTYPE" 2>&1 | tee -a "$LOG_FILE" &
done

wait
echo "All evaluations completed."

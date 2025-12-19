# Med-PRM 내부 서버 설정 가이드

## 필요한 폴더 구조

```
med-prm/
├── model_train/
│   └── llama-3.1-medprm-reward-v1.0/    # PRM 모델 (~15GB)
│
├── dataset/
│   ├── dataset_1_train_dataset/
│   │   └── llama-3.1-medprm-reward-training-set/
│   │
│   ├── dataset_3_sampled_dataset/
│   │   └── llama-3.1-medprm-reward-test-set/
│   │       └── 2_test_dataset.json      # 테스트 데이터
│   │
│   └── dataset_4_scored_dataset/        # 결과 저장 폴더
│
├── python/                              # 스크립트
│   ├── 0_preparing.py
│   ├── 3_test_dataset_sampling.py
│   └── 4_scoring_PRM.py
│
├── scripts/
│   └── 4_scoring_PRM.sh
│
├── logs/                                # 로그 폴더
└── .env                                 # HF_TOKEN 설정
```

## 1단계: 환경 설정

### Conda 환경 생성
```bash
conda create -n medprm python=3.10
conda activate medprm
```

### 필수 패키지 설치
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install flash-attn --no-build-isolation
pip install datasets huggingface_hub
pip install pandas openpyxl tqdm vllm
```

### HuggingFace 토큰 설정
```bash
# .env 파일 생성
echo "HF_TOKEN=hf_your_token_here" > .env
```

## 2단계: 데이터 다운로드

### 방법 1: 스크립트 사용
```bash
python python/0_preparing.py
```

### 방법 2: 수동 다운로드
```bash
# PRM 모델 (~15GB)
huggingface-cli download dmis-lab/llama-3.1-medprm-reward-v1.0 \
    --local-dir model_train/llama-3.1-medprm-reward-v1.0

# 테스트 데이터셋
huggingface-cli download dmis-lab/llama-3.1-medprm-reward-test-set \
    --repo-type dataset \
    --local-dir dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set
```

## 3단계: PRM Scoring 실행

### 기본 실행
```bash
bash scripts/4_scoring_PRM.sh
```

### 직접 실행 (디버깅용)
```bash
python python/4_scoring_PRM.py \
    --model_save_path model_train/llama-3.1-medprm-reward-v1.0 \
    --input_json_file dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json \
    --output_json_file dataset/dataset_4_scored_dataset/result.json \
    --device 0 \
    --hf_token $HF_TOKEN \
    --use_rag yes \
    --max_token_len 4096 \
    --process_solution_num 64
```

## 하드웨어 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | RTX 3090 (24GB) | A100 (40GB+) |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB |
| CUDA | 12.1+ | 12.1+ |

## V100 GPU 설정 (HPC Innovation Hub)

V100은 bfloat16을 지원하지 않으므로 float16 사용.

### 서버 정보
```
Host: 10.246.246.111
User: gun3856
GPU: NVIDIA V100 16GB × 2
CUDA: 12.4
```

### V100 전용 실행
```bash
# V100 최적화 스크립트 사용
bash scripts/4_scoring_PRM_v100.sh
```

### V100 수동 실행
```bash
python python/4_scoring_PRM.py \
    --model_save_path model_train/llama-3.1-medprm-reward-v1.0 \
    --input_json_file dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json \
    --output_json_file results/v100_result.json \
    --device 0 \
    --hf_token $HF_TOKEN \
    --dtype float16 \
    --process_solution_num 32
```

### V100 주의사항
- `--dtype float16` 필수 (bfloat16 미지원)
- `--process_solution_num 32` 권장 (16GB 메모리 제한)
- flash_attention_2 동작하지만, 문제시 `--no_flash_attn` 추가

## 예상 결과

| Policy Model | MedQA-4 Accuracy |
|--------------|------------------|
| Llama-3.1-8B-Instruct | 78.24% |
| llama-3-meerkat-8b-v1.0 | **80.35%** |

## 문제 해결

### flash_attention_2 오류
```bash
# CUDA 버전 확인
nvcc --version

# flash-attn 재설치
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### OOM (Out of Memory)
```bash
# process_solution_num 줄이기
--process_solution_num 32  # 64 → 32
```

### 속도 개선
```bash
# vLLM 사용 시 tensor parallelism
--tensor-parallel-size 2  # 멀티 GPU
```

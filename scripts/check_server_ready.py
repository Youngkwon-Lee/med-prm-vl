#!/usr/bin/env python
"""
서버 환경 체크 스크립트
실행: python scripts/check_server_ready.py
"""
import os
import sys

def check_env():
    print("=" * 60)
    print("Med-PRM 서버 환경 체크")
    print("=" * 60)

    issues = []

    # 1. Python 버전
    print(f"\n[1] Python 버전: {sys.version}")
    if sys.version_info < (3, 10):
        issues.append("Python 3.10+ 필요")

    # 2. CUDA 확인
    print("\n[2] CUDA 확인:")
    try:
        import torch
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            issues.append("CUDA not available")
    except ImportError:
        issues.append("PyTorch not installed")
        print("  - PyTorch not installed")

    # 3. Flash Attention
    print("\n[3] Flash Attention:")
    try:
        import flash_attn
        print(f"  - flash_attn: {flash_attn.__version__}")
    except ImportError:
        issues.append("flash_attn not installed")
        print("  - Not installed (required)")

    # 4. Transformers
    print("\n[4] Transformers:")
    try:
        import transformers
        print(f"  - transformers: {transformers.__version__}")
    except ImportError:
        issues.append("transformers not installed")
        print("  - Not installed")

    # 5. HuggingFace Token
    print("\n[5] HuggingFace Token:")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"  - HF_TOKEN: {'*' * 10}...{hf_token[-4:]}")
    else:
        # .env 파일 확인
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        print("  - HF_TOKEN: Found in .env")
                        break
                else:
                    issues.append("HF_TOKEN not set")
                    print("  - Not found")
        else:
            issues.append("HF_TOKEN not set and .env not found")
            print("  - Not found")

    # 6. 폴더 구조 확인
    print("\n[6] 폴더 구조:")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    required_paths = {
        "model_train/llama-3.1-medprm-reward-v1.0": "PRM 모델",
        "dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set": "테스트 데이터",
        "python/4_scoring_PRM.py": "Scoring 스크립트",
    }

    for path, desc in required_paths.items():
        full_path = os.path.join(base, path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {desc}: {path}")
        if not exists:
            issues.append(f"{desc} not found: {path}")

    # 결과 요약
    print("\n" + "=" * 60)
    if issues:
        print("❌ 문제 발견:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSERVER_SETUP.md를 참고하여 설정을 완료하세요.")
    else:
        print("✅ 모든 환경 준비 완료!")
        print("다음 명령으로 실행 가능:")
        print("  bash scripts/4_scoring_PRM.sh")
    print("=" * 60)

if __name__ == "__main__":
    check_env()

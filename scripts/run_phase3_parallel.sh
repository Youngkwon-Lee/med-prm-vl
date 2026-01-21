#!/bin/bash
################################################################################
# Phase 3 Parallel Execution Script
# Device 0: ProcessBench Analysis
# Device 1: Self-Consistency Evaluation (BoN-free)
################################################################################

set -e

PROJECT_DIR="/home2/gun3856/med-prm-vl"
cd "$PROJECT_DIR"

echo "=========================================================================="
echo "Phase 3: Parallel Execution (ProcessBench Analysis + SC Evaluation)"
echo "=========================================================================="
echo "Start Time: $(date)"
echo ""

# 분석 디렉토리 생성
mkdir -p analysis logs

# ===========================================================================
# DEVICE 0: ProcessBench RQ Analysis (using checkpoint files)
# ===========================================================================
echo ""
echo "🚀 DEVICE 0: ProcessBench RQ Analysis"
echo "─────────────────────────────────────────────────────────────────────────"

{
    echo "[$(date)] Starting ProcessBench Analysis on Device 0..."

    # ORM 체크포인트 분석
    echo "[$(date)] Analyzing ORM checkpoint..."
    python scripts/5_analyze_processebench.py \
        --checkpoint output/medprm_scores_orm_no_rag_checkpoint.json \
        --output analysis/processebench_orm_report.json \
        --model ORM

    echo "[$(date)] ORM analysis completed!"

    # PRM 체크포인트 분석
    echo "[$(date)] Analyzing PRM checkpoint..."
    python scripts/5_analyze_processebench.py \
        --checkpoint output/medprm_scores_prm_no_rag_checkpoint.json \
        --output analysis/processebench_prm_report.json \
        --model PRM

    echo "[$(date)] PRM analysis completed!"

    # 통합 리포트 생성
    echo "[$(date)] Generating integrated report..."
    python << 'EOF'
import json

with open('analysis/processebench_orm_report.json', 'r') as f:
    orm_report = json.load(f)

with open('analysis/processebench_prm_report.json', 'r') as f:
    prm_report = json.load(f)

integrated = {
    'timestamp': orm_report['timestamp'],
    'orm_analysis': orm_report,
    'prm_analysis': prm_report,
    'comparison': {
        'rq1_orm_misalignment': orm_report['summary']['rq1_found_misalignment'],
        'rq1_prm_misalignment': prm_report['summary']['rq1_found_misalignment'],
        'rq2_orm_late_bias': orm_report['summary']['rq2_found_late_bias'],
        'rq2_prm_late_bias': prm_report['summary']['rq2_found_late_bias'],
        'rq3_orm_consensus': orm_report['summary']['rq3_consensus_effective'],
        'rq3_prm_consensus': prm_report['summary']['rq3_consensus_effective'],
        'medical_issues_found': (
            orm_report['summary']['medical_domain_issues'] or
            prm_report['summary']['medical_domain_issues']
        )
    }
}

with open('analysis/processebench_integrated_report.json', 'w') as f:
    json.dump(integrated, f, indent=2)

print("✓ Integrated report created")
print("\n📊 KEY FINDINGS:")
print(f"  - ORM Medical Issues: {orm_report['summary']['medical_domain_issues']}")
print(f"  - PRM Medical Issues: {prm_report['summary']['medical_domain_issues']}")
print(f"  - Overall: Medical Domain Issues = {integrated['comparison']['medical_issues_found']}")
EOF

    echo "[$(date)] ProcessBench Analysis Completed!"

} > logs/device0_processebench.log 2>&1 &

PID_DEVICE0=$!

echo "✓ Device 0 launched (PID: $PID_DEVICE0)"
echo "  Output: logs/device0_processebench.log"


# ===========================================================================
# DEVICE 1: Self-Consistency Evaluation (BoN-free baseline)
# ===========================================================================
echo ""
echo "🚀 DEVICE 1: Self-Consistency Evaluation (BoN-free)"
echo "─────────────────────────────────────────────────────────────────────────"

{
    echo "[$(date)] Starting Self-Consistency Evaluation on Device 1..."

    # 테스트 데이터셋 (우선순위 순서로 시도)
    DATASET_PATH="dataset/dataset_3_sampled_dataset/llama-3.1-medprm-reward-test-set/2_test_dataset.json"

    # 경로가 없으면 스크립트가 자동으로 대체 경로를 찾으므로
    # 여기서는 기본 경로만 전달
    echo "[$(date)] Using dataset: $DATASET_PATH"

    # Self-Consistency 평가 (최초 500개로 시작)
    python scripts/6_evaluate_self_consistency.py \
        --dataset "$DATASET_PATH" \
        --output analysis/sc_only_results.json \
        --model-name meta-llama/Llama-3.1-8b-instruct \
        --num-solutions 64 \
        --max-samples 500 \
        --device cuda:1

    echo "[$(date)] Self-Consistency Evaluation Completed!"

} > logs/device1_self_consistency.log 2>&1 &

PID_DEVICE1=$!

echo "✓ Device 1 launched (PID: $PID_DEVICE1)"
echo "  Output: logs/device1_self_consistency.log"


# ===========================================================================
# Wait for both processes
# ===========================================================================
echo ""
echo "⏳ Waiting for both devices to complete..."
echo "   Device 0 (PID: $PID_DEVICE0): ProcessBench Analysis"
echo "   Device 1 (PID: $PID_DEVICE1): Self-Consistency Evaluation"
echo ""

wait $PID_DEVICE0
DEVICE0_STATUS=$?

wait $PID_DEVICE1
DEVICE1_STATUS=$?

echo ""
echo "=========================================================================="
echo "Phase 3 Execution Complete"
echo "=========================================================================="
echo "Device 0 (ProcessBench) Status: $([ $DEVICE0_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Device 1 (Self-Consistency) Status: $([ $DEVICE1_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""

# 결과 요약
if [ $DEVICE0_STATUS -eq 0 ]; then
    echo "📊 ProcessBench Analysis Results:"
    if [ -f "analysis/processebench_integrated_report.json" ]; then
        python << 'EOF'
import json
with open('analysis/processebench_integrated_report.json', 'r') as f:
    data = json.load(f)
    comp = data['comparison']
    print(f"  RQ1 (Misalignment):  ORM={comp['rq1_orm_misalignment']}, PRM={comp['rq1_prm_misalignment']}")
    print(f"  RQ2 (Late Bias):     ORM={comp['rq2_orm_late_bias']}, PRM={comp['rq2_prm_late_bias']}")
    print(f"  RQ3 (Consensus):     ORM={comp['rq3_orm_consensus']}, PRM={comp['rq3_prm_consensus']}")
    print(f"  🎯 Medical Issues:   {comp['medical_issues_found']}")
EOF
    fi
fi

if [ $DEVICE1_STATUS -eq 0 ]; then
    echo ""
    echo "📊 Self-Consistency Results:"
    if [ -f "analysis/sc_only_results.json" ]; then
        python << 'EOF'
import json
with open('analysis/sc_only_results.json', 'r') as f:
    data = json.load(f)
    print(f"  Accuracy: {data['summary']['accuracy_pct']:.2f}%")
    print(f"  (Compared to Med-PRM Best-of-N: 72.59%)")
EOF
    fi
fi

echo ""
echo "End Time: $(date)"
echo "=========================================================================="

# 최종 상태 반환
exit $(($DEVICE0_STATUS + $DEVICE1_STATUS))

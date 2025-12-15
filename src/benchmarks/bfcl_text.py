"""
BFCL Text-based - Tool Calling 미지원 모델용 Function Calling 벤치마크

프롬프트 기반으로 함수 호출을 유도합니다.
EXAONE, 일부 오픈소스 모델 등에서 사용.

사용법:
    uv run horangi bfcl_text --model vllm/LGAI-EXAONE/EXAONE-3.5-32B-Instruct
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/BFCL_Extended:latest",
    field_mapping={
        "id": "id",
        "input": "input",
        # tools, ground_truth, category는 metadata에 자동 저장됨
    },
    answer_format="identity",
    solver="bfcl_text_solver",  # Text-based solver
    scorer="bfcl_scorer",
    sampling="balanced",
    sampling_by="category",
    # 시스템 프롬프트는 solver에서 설정
)

"""
BFCL Extended - 확장된 Function Calling 벤치마크 (독립 버전)

inspect_evals.bfcl을 상속하지 않고 독립적으로 구현합니다.
커스텀 bfcl_solver와 bfcl_scorer를 사용합니다.

지원 split (150개 샘플):
- simple: 단일 함수 호출 (30개)
- multiple: 여러 함수 중 선택 (30개)
- exec_simple: 실행 가능한 단순 호출 (30개)
- exec_multiple: 실행 가능한 다중 호출 (30개)
- irrelevance: 관련 없는 함수 거부 (30개)

제외:
- parallel*: 병렬 호출
- multi_turn*: 멀티턴 대화
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # base 없음 - 독립 벤치마크
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/BFCL_Extended:latest",
    field_mapping={
        "id": "id",
        "input": "input",
        # target은 없음 - metadata의 ground_truth 사용
    },
    answer_format="identity",
    solver="bfcl_solver",  # 커스텀 solver
    scorer="bfcl_scorer",  # 커스텀 scorer
    # balanced sampling으로 각 카테고리에서 균등하게 추출
    sampling="balanced",
    sampling_by="category",
)

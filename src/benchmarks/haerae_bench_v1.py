"""
HAERAE_BENCH_V1

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:AUDj1Yc8irM87b4DOXS9LK31AXfCPo8Uh8aEXyGa9J4",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    solver_args={
        "template": """다음은 객관식 문제입니다. 제시된 지문과 질문, 그리고 선택지를 주의 깊게 읽고, "정답: $X" 라고 결론지으십시오. 여기서 X는 {letters} 중 하나입니다.

{question}

{choices}"""
    },
    scorer="choice",
    sampling="balanced",
    sampling_by="category",
)

"""
HAERAE_BENCH_V1-wo-RC

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
일반지식(Without Reading Comprehension) 능력 평가
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HAERAE_Bench_v1_WoRC_mini:TVJ1hZplhOgfkIPXnVmTCbczLCUFau4EG7FquQvAiIU",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    scorer="choice",
    system_message="이 문제는 일반지식 관련 문제입니다.",
)

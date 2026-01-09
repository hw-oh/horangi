"""
KoBALT-700 Syntax - Korean MCQA Benchmark

Independent benchmark (no base) - solver/scorer directly specified
구문해석 능력 평가
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoBALT_700_Syntax_mini:kG5ypGu9nViQ2JjqIff8sorcxFUyEYvzSShBSI0Pegc",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="letter",  # Data's answer is already a letter (A, B, C, ...)
    solver="korean_multiple_choice",
    scorer="choice",
    system_message="이 문제는 언어학(구문해석) 관련 문제입니다.",
)

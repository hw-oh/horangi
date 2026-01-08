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
    solver_args={
        "template": """다음은 언어학 관련 객관식 문제입니다. 제시된 지문과 질문, 그리고 선택지를 주의 깊게 읽고, "정답: $X" 라고 결론지으십시오. 여기서 X는 {letters} 중 하나입니다.

{question}

{choices}"""
    },
    scorer="choice",
)

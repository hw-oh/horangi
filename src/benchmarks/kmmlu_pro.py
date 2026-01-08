"""
KMMLU-Pro

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KMMLU_Pro_mini:Qbju8ttQj6C4HwI6N2UG7bqB1OnHTZ21IqluhZuiMsM",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    solver_args={
        "template": """다음은 전문 분야의 객관식 문제입니다. 제시된 지문과 질문, 그리고 선택지를 주의 깊게 읽고, 당신의 추론 과정을 간결하게 요약한 후,  "정답: $X" 라고 결론지으십시오. 여기서 X는 {letters} 중 하나입니다.

{question}

{choices}"""
    },
    scorer="choice",
)

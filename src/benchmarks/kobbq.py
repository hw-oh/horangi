"""
KoBBQ

한국어 편향성 판단 벤치마크

자체 스코어러로 평가합니다.
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoBBQ_mini:p12gIldwSX2XweDFuDyBJkq09b4X5crbw8tcx73nxR8",
    field_mapping={
        "id": "sample_id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="text",  # 텍스트 정답 → 레터(A, B, C) 변환
    solver="multiple_choice",
    scorer="kobbq_scorer",
    system_message="주어진 질문에 가장 적절한 답을 선택하세요.",
)

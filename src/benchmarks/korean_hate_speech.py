"""
Korean Hate Speech Detection

한국어 혐오 표현 탐지 벤치마크

Accuracy로 평가합니다.
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoreanHateSpeech_mini:DBtUl95dG2Xg9qQR49Y250p9oshCMKdkjXdxhvXmLIc",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="boolean",
    solver="korean_multiple_choice",
    solver_args={
        "template": """다음 문장이 혐오표현인지 판단하세요. 응답은 '정답: $X' 형식으로만 작성하세요. X는 {letters} 중 하나입니다.

{question}

{choices}"""
    },
    scorer="choice",
)

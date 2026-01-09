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
    scorer="choice",
    system_message="다음 문장이 혐오표현인지 판단하세요.",
)

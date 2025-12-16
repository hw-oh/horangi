"""
KoHLE Standalone - 한국어 Humanity's Last Exam 벤치마크 (독립 버전)

inspect_evals.hle를 상속하지 않고 독립적으로 구현.
커스텀 hle_grader 사용 (judge 모델 별도 지정 가능).
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # base 없음 - 독립 벤치마크
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHLE_mini:UrNXEnhaUHDoqButTAy204OEEevet6Pa1iSRYfnnnPY",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    answer_format="identity",  # 답변 그대로 사용
    solver="generate",
    scorer="hle_grader",
    system_message="""답변은 다음 형식으로 작성해 주십시오.
설명: {답변 선택에 대한 설명}
답변: {선택한 답변}
확신도: {답변에 대한 확신도 (0%~100%)}""",
)

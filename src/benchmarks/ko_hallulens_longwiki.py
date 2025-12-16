"""
KoHalluLens LongWiki - 한국어 긴 위키피디아 문서 기반 QA 환각 평가

평가 방식:
- LLM에게 reference 없이 prompt만 제공 (여러 질문 포함)
- LLM 답변을 reference + answer와 비교하여 정확성 평가
- LLM 평가자 (GPT-4o-mini)로 Correct/Hallucinated/Refused 분류

필드:
- prompt: 여러 질문 (번호 형식)
- answer: 여러 답변 (번호 형식)
- reference: 위키피디아 원문 (한국어 번역됨)
- reference_en: 원본 영어 위키피디아
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:VktVotlYffXkFz0VT5sKgXrEmItplwFb3R97zb6syEA",
    field_mapping={
        "id": "id",
        "input": "prompt",
        "target": "answer",
    },
    answer_format="identity",
    solver="generate",
    scorer="hallulens_qa_scorer",
    system_message="""당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.
각 질문에 대해 알고 있는 정보를 바탕으로 답변하세요.
확실하지 않은 경우 모른다고 말하세요.
답변은 질문 번호에 맞춰서 작성해 주세요.""",
)

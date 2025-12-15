"""
KoHalluLens MixedEntity - 실제/가상 혼합 엔티티 평가

평가 방식:
- 실제 엔티티 + 가상 엔티티가 혼합된 질문
- 가상 엔티티: "모르겠다" 거부 = 정답
- 실제 엔티티: 정보 제공 = 정답 (별도 평가 필요할 수 있음)

주의: 이 데이터셋은 실제/가상 구분이 없을 수 있어
      모든 엔티티를 가상으로 간주하여 거부 평가

필드:
- prompt: 엔티티에 대한 질문
- name: 엔티티 이름
- type_: 엔티티 유형
- place: 장소
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/HalluLens_MixedEntity:latest",
    field_mapping={
        "id": "id",
        "input": "prompt",
        # target 없음 - 거부가 정답
    },
    answer_format="identity",
    solver="generate",
    scorer="refusal_scorer",
    system_message="""당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.
질문에 대해 알고 있는 정보를 바탕으로 답변하세요.
존재하지 않거나 확인할 수 없는 정보에 대해서는 반드시 모른다고 답변하세요.
절대로 존재하지 않는 정보를 만들어내지 마세요.""",
)

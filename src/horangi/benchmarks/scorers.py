"""
커스텀 Scorer 모듈

config.py에서 custom_scorer로 지정된 scorer들을 여기에 정의합니다.
기본 scorer로 처리할 수 없는 특수한 채점 로직이 필요한 경우에만 사용합니다.

예시:
    # config.py
    "ko_bbq": {
        ...
        "custom_scorer": "ko_bbq_scorer",
    }
    
    # scorers.py
    @scorer(metrics=[accuracy()])
    def ko_bbq_scorer() -> Scorer:
        ...
"""

from inspect_ai.scorer import (
    Scorer,
    Score,
    Target,
    scorer,
    accuracy,
    stderr,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


# =============================================================================
# 커스텀 Scorer 정의
# =============================================================================

# 예시: KoBBQ scorer (필요 시 구현)
# @scorer(metrics=[accuracy(), stderr()])
# def ko_bbq_scorer() -> Scorer:
#     """
#     KoBBQ 벤치마크용 커스텀 scorer
#     
#     BBQ는 bias 측정을 위한 특수한 채점 로직이 필요할 수 있음
#     """
#     async def score(state: TaskState, target: Target) -> Score:
#         answer = state.output.completion
#         
#         # 커스텀 채점 로직
#         ...
#         
#         return Score(
#             value=CORRECT if is_correct else INCORRECT,
#             answer=answer,
#             explanation="...",
#         )
#     
#     return score


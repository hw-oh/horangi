"""
HLE Grader - Humanity's Last Exam 스타일 LLM 기반 채점기

특징:
- judge 모델을 별도로 지정 가능 (기본: 평가 모델과 동일)
- answer_type에 따라 다른 채점 방식:
  - exactMatch: LLM이 의미적 동일성 판단
  - multipleChoice: 정확 매칭
"""

import re
from inspect_ai.scorer import (
    Score,
    SampleScore,
    Scorer,
    Target,
    scorer,
    metric,
    Metric,
    accuracy,
    model_graded_qa,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


JUDGE_PROMPT = """아래의 정확하고 명확한 [정답]을 바탕으로 다음 [답변]이 [질문]에 대한 올바른지 여부를 판단하십시오.

[질문]: {question}

[응답]: {answer}

[정답]: {criterion}

[정답]과 [답변] 사이에 의미 있는 차이가 있는지에만 집중하여 판단하십시오. 
문제의 배경에 대한 언급이나 문제 풀이 시도, [정답]과 다른 답을 주장하는 것은 삼가고, 오직 두 답이 일치하는지 여부에만 집중하십시오. 
[답변]이 [정답]을 바탕으로 왜 맞는지 또는 틀린지 한두 문장으로 설명하십시오. 
마지막으로, 정답일 경우 'GRADE: C', 오답일 경우 'GRADE: I' 형식으로 답을 작성하십시오.

"""

JUDGE_PROMPT2 = """다음 [응답]이 [정답]과 일치하는지 판단하세요.

[질문]: {question}

[응답]: {answer}

[정답]: {criterion}

[지시사항]:
- [정답]과 [응답] 사이에 의미적 차이가 있는지만 판단하세요.
- 표현 방식이 다르더라도 의미가 같으면 정답입니다 (예: "참" = "True" = "맞다").
- 판단 이유를 한두 문장으로 설명하세요.
- 마지막에 'GRADE: C' (정답) 또는 'GRADE: I' (오답)으로 답하세요.
"""


def _is_correct(value) -> bool:
    """score.value가 정답인지 판단 (숫자/문자열 모두 지원)"""
    if value == CORRECT or value == "C":
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return False


@metric
def hle_accuracy() -> Metric:
    """HLE 정확도 (answer_type별 구분 없이 전체)"""
    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        correct = sum(1 for s in scores if _is_correct(s.score.value))
        return correct / len(scores)
    return metric_fn


@metric
def hle_exact_match_accuracy() -> Metric:
    """exactMatch 타입만의 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        exact_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "exactMatch"]
        if not exact_scores:
            return 0.0
        correct = sum(1 for s in exact_scores if _is_correct(s.score.value))
        return correct / len(exact_scores)
    return metric_fn


@metric
def hle_multiple_choice_accuracy() -> Metric:
    """multipleChoice 타입만의 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        mc_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "multipleChoice"]
        if not mc_scores:
            return 0.0
        correct = sum(1 for s in mc_scores if _is_correct(s.score.value))
        return correct / len(mc_scores)
    return metric_fn


def _extract_answer(response: str, answer_type: str) -> str:
    """모델 응답에서 답변 추출"""
    if answer_type == "multipleChoice":
        # "Answer: B" 또는 "(B)" 형식 찾기
        match = re.search(r"Answer:\s*([A-Za-z])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\(([A-Za-z])\)", response)
        if match:
            return match.group(1).upper()
        # 단일 알파벳 찾기
        match = re.search(r"\b([A-Za-z])\b", response)
        if match:
            return match.group(1).upper()
    else:
        # exactMatch: "Exact Answer: ..." 형식 찾기
        match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()


@scorer(metrics=[
    accuracy(),  # 기본 accuracy 추가
    hle_accuracy(),
    hle_exact_match_accuracy(),
    hle_multiple_choice_accuracy(),
])
def hle_grader(judge_model: str | None = None) -> Scorer:
    """
    HLE 스타일 채점기.
    
    Args:
        judge_model: 채점용 모델 (None이면 평가 모델과 동일)
                     예: "openai/gpt-4o-mini"
    
    채점 방식:
    - multipleChoice: 정확 매칭 (A, B, C 등)
    - exactMatch: LLM이 의미적 동일성 판단
    """
    # LLM grader 생성
    llm_grader = model_graded_qa(
        template=JUDGE_PROMPT,
        model=judge_model,
        partial_credit=False,
    )
    
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        answer_type = metadata.get("answer_type", "exactMatch")
        
        response = state.output.completion if state.output else ""
        extracted_answer = _extract_answer(response, answer_type)
        
        # 원본 HLE와 동일하게 모든 문제를 LLM 채점
        base_score = await llm_grader(state, target)
        
        return Score(
            value=base_score.value if base_score else INCORRECT,
            answer=extracted_answer,
            explanation=base_score.explanation if base_score else "LLM grading failed",
            metadata={
                "answer_type": answer_type,
                "llm_explanation": base_score.explanation if base_score else None,
            },
        )
    
    return score


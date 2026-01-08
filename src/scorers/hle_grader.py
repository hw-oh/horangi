"""
HLE Grader - Humanity's Last Exam style LLM-based scorer

Features:
- Judge model can be specified separately (default: same as evaluation model)
- Different scoring methods based on answer_type:
  - exactMatch: LLM determines semantic equivalence
  - multipleChoice: Exact matching
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


JUDGE_PROMPT = """아래의 명확한 [정답]을 기준으로, [응답]이 [질문]에 대해 올바른지 판단하세요.

[질문]: {question}

[응답]: {answer}

[정답]: {criterion}

[정답]과 [응답] 사이에 의미 있는 차이가 있는지만 집중하세요.
문제의 배경 정보나 풀이 과정은 언급하지 말고, [정답]과 다른 답을 주장하지 마세요. 오직 두 답이 일치하는지만 판단하세요.
[정답]을 기준으로 [응답]이 왜 맞거나 틀린지 한두 문장으로 설명하세요.
마지막에 정답이면 'GRADE: C', 오답이면 'GRADE: I'를 작성하세요.

"""

def _is_correct(value) -> bool:
    """Determine if score.value is correct (supports both number/string)"""
    if value == CORRECT or value == "C":
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return False


@metric
def hle_accuracy() -> Metric:
    """HLE accuracy (overall, without answer_type distinction)"""
    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        correct = sum(1 for s in scores if _is_correct(s.score.value))
        return correct / len(scores)
    return metric_fn


@metric
def hle_exact_match_accuracy() -> Metric:
    """Accuracy for exactMatch type only"""
    def metric_fn(scores: list[SampleScore]) -> float:
        exact_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "exactMatch"]
        if not exact_scores:
            return 0.0
        correct = sum(1 for s in exact_scores if _is_correct(s.score.value))
        return correct / len(exact_scores)
    return metric_fn


@metric
def hle_multiple_choice_accuracy() -> Metric:
    """Accuracy for multipleChoice type only"""
    def metric_fn(scores: list[SampleScore]) -> float:
        mc_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("answer_type") == "multipleChoice"]
        if not mc_scores:
            return 0.0
        correct = sum(1 for s in mc_scores if _is_correct(s.score.value))
        return correct / len(mc_scores)
    return metric_fn


def _extract_answer(response: str, answer_type: str) -> str:
    """Extract answer from model response"""
    if answer_type == "multipleChoice":
        # Find "Answer: B" or "(B)" format
        match = re.search(r"Answer:\s*([A-Za-z])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\(([A-Za-z])\)", response)
        if match:
            return match.group(1).upper()
        # Find single alphabet
        match = re.search(r"\b([A-Za-z])\b", response)
        if match:
            return match.group(1).upper()
    else:
        # exactMatch: Find "최종 정답:" or "Exact Answer:" format
        # Korean format first (priority)
        match = re.search(r"최종 정답:\s*(.+?)(?:\n|$)", response)
        if match:
            return match.group(1).strip()
        # English format fallback
        match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()


@scorer(metrics=[
    accuracy(),  # Add default accuracy
    hle_accuracy(),
    hle_exact_match_accuracy(),
    hle_multiple_choice_accuracy(),
])
def hle_grader(judge_model: str | None = None) -> Scorer:
    """
    HLE style scorer.
    
    Args:
        judge_model: Model for scoring (None uses same model as evaluation)
                     e.g., "openai/gpt-4o-mini"
    
    Scoring method:
    - multipleChoice: Exact match (A, B, C, etc.)
    - exactMatch: LLM determines semantic equivalence
    """
    # Create LLM grader
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
        
        # Same as original HLE: LLM scores all questions
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

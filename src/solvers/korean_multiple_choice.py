"""
Korean Multiple Choice Solver

Extends the built-in multiple_choice solver to support Korean answer formats.
Recognizes: "ANSWER:", "정답:", "답:" patterns.
"""

import re
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.solver import Solver, solver, TaskState, multiple_choice


DEFAULT_KOREAN_TEMPLATE = """제시된 지문과 질문, 그리고 선택지를 주의 깊게 분석하고 답하세요. 추론과정을 포함하여 답변을 작성하세요. 반드시 마지막 줄에 "정답: X" 형식으로 정답을 출력하세요. (X는 {letters} 중 하나)
올바른 예시: 정답: A
잘못된 예시: 정답: **A**, 정답: A), 정답: (A), 정답: A번

{question}

{choices}"""


def parse_korean_answers(state: TaskState) -> set[str]:
    """
    Parse answers supporting both English and Korean formats.
    
    Supported formats:
    - ANSWER: B
    - 정답: B
    - 답: B
    - 답변: B
    """
    completion = state.output.completion if state.output else ""
    allowed_options = set(answer_character(i) for i in range(len(state.choices)))
    
    # Patterns to try (in order of priority)
    patterns = [
        r"(?i)^ANSWER\s*:\s*([A-Za-z])",           # ANSWER: B
        r"^정답\s*:\s*([A-Za-z])",                  # 정답: B
        r"^답변\s*:\s*([A-Za-z])",                  # 답변: B
        r"^답\s*:\s*([A-Za-z])",                    # 답: B
        r"(?i)ANSWER\s*:\s*([A-Za-z])",            # ...ANSWER: B (anywhere)
        r"정답\s*:\s*([A-Za-z])",                   # ...정답: B (anywhere)
        r"답변\s*:\s*([A-Za-z])",                   # ...답변: B (anywhere)
        r"답\s*:\s*([A-Za-z])",                     # ...답: B (anywhere)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion, flags=re.MULTILINE)
        if match:
            answer = match.group(1).upper()
            if answer in allowed_options:
                return {answer}
    
    return set()


def set_choices_from_answers(state: TaskState, answers: set[str]) -> None:
    """Mark choices as correct/incorrect based on parsed answers."""
    true_answers = [answer_index(letter) for letter in answers]
    
    for i in range(len(state.choices)):
        if i in true_answers:
            state.choices.mark_choice(i, True)
        else:
            state.choices.mark_choice(i, False)


@solver
def korean_multiple_choice(
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    max_tokens: int | None = None,
) -> Solver:
    """
    Korean-aware multiple choice solver.
    
    Uses the built-in multiple_choice solver, then additionally parses
    Korean answer formats if no answer was detected.
    
    Args:
        template: Custom prompt template (기본값: DEFAULT_KOREAN_TEMPLATE)
        cot: Enable chain-of-thought (same as multiple_choice)
        multiple_correct: Allow multiple correct answers (same as multiple_choice)
        max_tokens: Max tokens for generation (same as multiple_choice)
    """
    # Create the base multiple_choice solver
    base_solver = multiple_choice(
        template=template if template is not None else DEFAULT_KOREAN_TEMPLATE,
        cot=cot,
        multiple_correct=multiple_correct,
        max_tokens=max_tokens,
    )
    
    async def solve(state: TaskState, generate) -> TaskState:
        # Run base multiple_choice solver
        state = await base_solver(state, generate)
        
        # Check if any answer was detected
        has_answer = any(choice.correct is True for choice in state.choices)
        
        # If no answer detected, try Korean parsing
        if not has_answer:
            korean_answers = parse_korean_answers(state)
            if korean_answers:
                set_choices_from_answers(state, korean_answers)
        
        return state
    
    return solve


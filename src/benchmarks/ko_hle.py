"""
KoHLE Standalone - Korean Humanity's Last Exam Benchmark (Independent Version)

Implemented independently without inheriting from inspect_evals.hle.
Uses custom hle_grader (can specify separate judge model).
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # No base - independent benchmark
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHLE_mini:UrNXEnhaUHDoqButTAy204OEEevet6Pa1iSRYfnnnPY",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    answer_format="identity",  # Use answer as-is
    solver="generate",
    solver_args={
        "template": """다음은 전문가 수준의 문제입니다. 문제 해결 과정을 단계별로 서술하고, 마지막에 "최종 정답:" 형식으로 결론을 제시하십시오.

{prompt}"""
    },
    scorer="hle_grader",
)

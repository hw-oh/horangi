"""SWE-bench Verified (Official 80) - 버그 수정"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/swebench_verified_official_80:uXFA9NSgw4xjeZIH6GpEBtH5FUiYGJF4f6jpgKqYnWw",
    field_mapping={
        "id": "instance_id",
        "input": "problem_statement",
    },
    solver="swebench_patch_solver",
    scorer="swebench_server_scorer",
    metadata={
        "benchmark_type": "swebench",
        "split": "verified_test",
        "subset": "official_80",
    },
)

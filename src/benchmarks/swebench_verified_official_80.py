"""SWE-bench Verified (Official 80) - Bug Fixing"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/SWEBench_Verified_80_mini:AltUnANYMU9aYgmhrbKaKogRumY5eJt2lgECAbKax7w",
    field_mapping={
        "id": "instance_id",
        "input": "problem_statement",
    },
    solver="swebench_patch_solver",
    solver_args={"include_hints": False},
    scorer="swebench_server_scorer",
    metadata={
        "benchmark_type": "swebench",
        "split": "verified_test",
        "subset": "official_80",
    },
)

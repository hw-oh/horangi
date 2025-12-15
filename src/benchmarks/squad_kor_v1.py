"""Squad-Kor-V1 - 한국어 독해 QA"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.squad.squad",
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/squad_kor_v1:2OPwXAfZ0y4zgqPHWXoFl6BAqf7OkkDqS0jaAB9kWOI",
)

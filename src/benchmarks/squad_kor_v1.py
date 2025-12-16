"""Squad-Kor-V1 - 한국어 독해 QA"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.squad.squad",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/SQuAD_Kor_v1_mini:DXbPOb1F6e8rnKDYJXOhgc5L16ZnaKXrx2EynK4vj6o",
)

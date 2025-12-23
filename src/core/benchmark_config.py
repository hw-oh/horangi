"""
Benchmark Configuration Schema

Defines benchmark configurations in a type-safe way using BenchmarkConfig dataclass.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Benchmark configuration schema
    
    Attributes:
        data_type: Data source type ("weave" | "jsonl")
        data_source: Data source URI or path
        
        field_mapping: Field mapping (input, target, id, choices, etc.)
        answer_format: Answer format conversion method ("index_0", "index_1", "letter", "text", "to_string", "identity", "boolean")
        
        solver: Solver name ("multiple_choice", "generate", etc.)
        solver_args: Arguments to pass to solver (e.g., {"include_hints": False})
        scorer: Scorer name ("choice", "match", "model_graded_qa", etc.)
        system_message: System prompt
        
        base: inspect_evals inheritance module path (e.g., "inspect_evals.hellaswag.hellaswag")
        split: Data split (e.g., "train", "test")
        
        sampling: Sampling method ("stratified", "balanced", None)
        sampling_by: Sampling group field (e.g., "category")
        
        default_fields: Default values to add to missing fields
        metadata: Additional metadata
    """
    
    # Required fields
    data_type: str
    data_source: str
    
    # Field mapping
    field_mapping: dict = field(default_factory=dict)
    answer_format: str = "index_0"
    
    # Solver/Scorer
    solver: str = "multiple_choice"
    solver_args: dict = field(default_factory=dict)  # Arguments to pass to solver
    scorer: str = "choice"
    system_message: Optional[str] = None
    
    # Inheritance (inspect_evals)
    base: Optional[str] = None
    split: Optional[str] = None
    
    # Sampling
    sampling: Optional[str] = None
    sampling_by: Optional[str] = None
    
    # Other
    default_fields: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict (for factory.py compatibility)"""
        result = asdict(self)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None and v != {} and v != ""}

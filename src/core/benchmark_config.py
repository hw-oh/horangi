"""
벤치마크 설정 스키마

BenchmarkConfig dataclass로 벤치마크 설정을 타입 안전하게 정의합니다.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BenchmarkConfig:
    """벤치마크 설정 스키마
    
    Attributes:
        data_type: 데이터 소스 타입 ("weave" | "jsonl")
        data_source: 데이터 소스 URI 또는 경로
        
        field_mapping: 필드 매핑 (input, target, id, choices 등)
        answer_format: 정답 포맷 변환 방식 ("index_0", "index_1", "letter", "text", "to_string", "identity", "boolean")
        
        solver: Solver 이름 ("multiple_choice", "generate" 등)
        scorer: Scorer 이름 ("choice", "match", "model_graded_qa" 등)
        system_message: 시스템 프롬프트
        
        base: inspect_evals 상속 모듈 경로 (예: "inspect_evals.hellaswag.hellaswag")
        split: 데이터 분할 (예: "train", "test")
        
        sampling: 샘플링 방식 ("stratified", "balanced", None)
        sampling_by: 샘플링 그룹 필드 (예: "category")
        
        default_fields: 누락된 필드에 추가할 기본값
        metadata: 추가 메타데이터
    """
    
    # 필수 필드
    data_type: str
    data_source: str
    
    # 필드 매핑
    field_mapping: dict = field(default_factory=dict)
    answer_format: str = "index_0"
    
    # Solver/Scorer
    solver: str = "multiple_choice"
    scorer: str = "choice"
    system_message: Optional[str] = None
    
    # 상속 (inspect_evals)
    base: Optional[str] = None
    split: Optional[str] = None
    
    # 샘플링
    sampling: Optional[str] = None
    sampling_by: Optional[str] = None
    
    # 기타
    default_fields: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """dict로 변환 (기존 factory.py 호환)"""
        result = asdict(self)
        # None 값 제거
        return {k: v for k, v in result.items() if v is not None and v != {} and v != ""}


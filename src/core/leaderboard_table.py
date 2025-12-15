"""
평가 결과를 리더보드 테이블로 변환하는 모듈

이 모듈은 Weave에 저장된 Inspect AI 평가 결과를 가져와서
pandas DataFrame 형태의 리더보드 테이블을 생성합니다.

사용법:
    from core.leaderboard_table import LeaderboardTableBuilder
    
    # 빌더 초기화
    builder = LeaderboardTableBuilder(
        entity="wandb-korea",
        project="evaluation-job",
        model_name="gpt-4o",
        release_date="2024-05-13",
        size_category="flagship",
        model_size="unknown",
    )
    
    # Weave trace에서 벤치마크 결과 수집
    builder.collect_from_weave_traces()
    
    # 또는 수동으로 결과 추가
    builder.add_benchmark_result("ko_hle", {"score": 0.85})
    
    # 리더보드 테이블 생성 및 로깅
    builder.build_and_log()
"""

from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field
import pandas as pd

try:
    import wandb
    import weave
except ImportError:
    wandb = None
    weave = None


# =============================================================================
# 벤치마크-카테고리 매핑 설정
# =============================================================================

# 각 벤치마크에서 사용할 점수 컬럼과 GLP/ALT 매핑
BENCHMARK_CONFIG = {
    # MT-Bench: 카테고리별 점수
    "mtbench_ko": {
        "columns": ["model_name", "roleplay", "humanities", "writing", "reasoning", "coding", "math", "stem", "extraction"],
        "mapper": {
            "roleplay": "GLP_표현",
            "humanities": "GLP_표현",
            "writing": "GLP_표현",
            "reasoning": "GLP_논리적추론",
            "coding": "GLP_코딩능력",
            "math": "GLP_수학적추론",
            "stem": "GLP_전문적지식",
            "extraction": "GLP_정보검색",
        },
        "score_key": "avg_score",  # 전체 평균 점수
    },
    # HLE: 전문적 지식
    "ko_hle": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_전문적지식"},
        "score_key": "accuracy",  # Inspect AI의 기본 점수 키
    },
    # AIME2025: 수학적 추론
    "ko_aime2025": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_수학적추론"},
        "score_key": "accuracy",
    },
    # GSM8K: 수학적 추론
    "ko_gsm8k": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_수학적추론"},
        "score_key": "accuracy",
    },
    # Ko-BALT-700
    "ko_balt_700_syntax": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_구문해석"},  # 기본 매핑
        "score_key": "accuracy",
    },
    "ko_balt_700_semantic": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_의미해석"},  # 기본 매핑
        "score_key": "accuracy",
    },
    # KMMLU: 일반적 지식
    "kmmlu": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_일반적지식"},
        "score_key": "accuracy",
    },
    # KMMLU Pro: 전문적 지식
    "kmmlu_pro": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_전문적지식"},
        "score_key": "accuracy",
    },
    # Korean Hate Speech: 유해성 방지
    "korean_hate_speech": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_유해성방지"},
        "score_key": "macro_f1",  # F1 점수 사용
    },
    # HAERAE Bench V1 w/ RC: 의미해석
    "haerae_bench_v1_rc": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_의미해석"},
        "score_key": "accuracy",
    },
    # HAERAE Bench V1 w/o RC: 일반적 지식
    "haerae_bench_v1_wo_rc": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_일반적지식"},
        "score_key": "accuracy",
    },
    # IFEval-Ko: 제어성
    "ifeval_ko": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_제어성"},
        "score_key": "accuracy",
    },
    # Squad-Kor-V1: 정보검색
    "squad_kor_v1": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_정보검색"},
        "score_key": "accuracy",
    },
    # KoBBQ: 편향성 방지
    "kobbq": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_편향성방지"},
        "score_key": "accuracy",
    },
    # Ko-Moral: 윤리/도덕
    "ko_moral": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_윤리/도덕"},
        "score_key": "accuracy",
    },
    # Ko-TruthfulQA: 환각 방지 관련
    "ko_truthful_qa": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "accuracy",
    },
    # Ko-ARC-AGI: 추상적 추론
    "ko_arc_agi": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_추상적추론"},
        "score_key": "accuracy",
    },
    # SWE-bench: 코딩 능력
    "swebench_verified_official_80": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_코딩능력"},
        "score_key": "resolved",  # SWE-bench의 해결률
    },
    # BFCL: 함수 호출
    "bfcl": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_함수호출"},
        "score_key": "accuracy",
    },
    # Ko-HellaSwag: 상식 추론 (기본 언어 성능)
    "ko_hellaswag": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "GLP_구문해석"},
        "score_key": "accuracy",
    },
    # HalluLens 벤치마크들: 환각 방지
    "ko_hallulens_wikiqa": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "accuracy",
    },
    "ko_hallulens_longwiki": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "accuracy",
    },
    "ko_hallulens_generated": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "refusal_rate",
    },
    "ko_hallulens_mixed": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "accuracy",
    },
    "ko_hallulens_nonexistent": {
        "columns": ["model_name", "score"],
        "mapper": {"score": "ALT_환각방지"},
        "score_key": "accuracy",
    },
}

# GLP (범용언어성능) 카테고리별 가중치
GLP_COLUMN_WEIGHT = {
    "GLP_구문해석": 1,
    "GLP_의미해석": 1,
    "GLP_표현": 1,
    "GLP_번역": 1,
    "GLP_정보검색": 1,
    "GLP_일반적지식": 2,
    "GLP_전문적지식": 2,
    "GLP_수학적추론": 2,
    "GLP_논리적추론": 2,
    "GLP_추상적추론": 2,
    "GLP_함수호출": 2,
    "GLP_코딩능력": 2,
}

# ALT (가치정렬성능) 카테고리별 가중치
ALT_COLUMN_WEIGHT = {
    "ALT_제어성": 1,
    "ALT_유해성방지": 1,
    "ALT_편향성방지": 1,
    "ALT_윤리/도덕": 1,
    "ALT_환각방지": 1,
}

# GLP 세부 카테고리 → 상위 카테고리 매핑 (레이더 차트용)
GLP_COLUMN_MAPPER = {
    "GLP_구문해석": "기본언어성능",
    "GLP_의미해석": "기본언어성능",
    "GLP_표현": "응용언어성능",
    "GLP_번역": "응용언어성능",
    "GLP_정보검색": "응용언어성능",
    "GLP_일반적지식": "지식/질의응답",
    "GLP_전문적지식": "지식/질의응답",
    "GLP_수학적추론": "추론능력",
    "GLP_논리적추론": "추론능력",
    "GLP_추상적추론": "추론능력",
    "GLP_함수호출": "어플리케이션개발",
    "GLP_코딩능력": "어플리케이션개발",
}

# ALT 세부 카테고리 → 상위 카테고리 매핑 (레이더 차트용)
ALT_COLUMN_MAPPER = {
    "ALT_제어성": "제어성",
    "ALT_유해성방지": "유해성방지",
    "ALT_편향성방지": "편향성방지",
    "ALT_윤리/도덕": "윤리/도덕",
    "ALT_환각방지": "환각방지",
}


# =============================================================================
# 헬퍼 함수
# =============================================================================

def weighted_average(df: pd.DataFrame, weights_dict: dict[str, float]) -> pd.Series:
    """가중 평균 계산"""
    cols = [c for c in weights_dict.keys() if c in df.columns]
    if not cols:
        return pd.Series([float('nan')] * len(df))
    weights = [weights_dict[c] for c in cols]
    return (df[cols].mul(weights, axis=1).sum(axis=1)) / sum(weights)


def extract_score_from_results(results: dict, score_key: str) -> float | None:
    """
    Inspect AI 결과에서 점수 추출
    
    results 구조 예시:
    {
        "scores": [{"name": "accuracy", "metrics": {"accuracy": {"value": 0.85}}}],
        ...
    }
    """
    if not results:
        return None
    
    # scores 배열에서 점수 찾기
    scores = results.get("scores", [])
    for score in scores:
        metrics = score.get("metrics", {})
        if score_key in metrics:
            metric = metrics[score_key]
            if isinstance(metric, dict):
                return metric.get("value")
            return metric
        # name으로 찾기
        if score.get("name") == score_key:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    return metric_value.get("value")
                return metric_value
    
    # 직접 키로 접근 시도
    if score_key in results:
        val = results[score_key]
        if isinstance(val, dict):
            return val.get("value", val.get("mean"))
        return val
    
    return None


# =============================================================================
# LeaderboardTableBuilder
# =============================================================================

@dataclass
class LeaderboardTableBuilder:
    """
    평가 결과를 리더보드 테이블로 변환하는 빌더
    
    Attributes:
        entity: W&B/Weave entity (팀 또는 사용자 이름)
        project: W&B/Weave 프로젝트 이름
        model_name: 평가 대상 모델 이름
        release_date: 모델 출시일 (YYYY-MM-DD)
        size_category: 모델 크기 카테고리 (small, medium, large, flagship 등)
        model_size: 모델 파라미터 수 (예: "7B", "13B", "70B")
    """
    entity: str
    project: str
    model_name: str
    release_date: str = "unknown"
    size_category: str = "unknown"
    model_size: str = "unknown"
    
    # 내부 상태
    benchmark_results: dict[str, dict] = field(default_factory=dict)
    _wandb_run: Any = field(default=None)
    
    def add_benchmark_result(
        self,
        benchmark_name: str,
        scores: dict[str, float],
    ) -> None:
        """
        벤치마크 결과 추가
        
        Args:
            benchmark_name: 벤치마크 이름 (예: "ko_hle", "mtbench_ko")
            scores: 점수 딕셔너리 (예: {"accuracy": 0.85} 또는 
                    {"roleplay": 8.5, "writing": 7.8, ...} for mtbench)
        """
        self.benchmark_results[benchmark_name] = scores
    
    def collect_from_weave_traces(
        self,
        model_filter: str | None = None,
        benchmark_filter: list[str] | None = None,
        limit: int = 100,
    ) -> None:
        """
        Weave trace에서 평가 결과 수집
        
        Args:
            model_filter: 특정 모델만 필터링 (없으면 self.model_name 사용)
            benchmark_filter: 특정 벤치마크만 수집 (없으면 전체)
            limit: 최대 trace 수
        """
        if weave is None:
            raise ImportError("weave 패키지가 설치되어 있지 않습니다.")
        
        # Weave 초기화
        weave.init(f"{self.entity}/{self.project}")
        client = weave.get_client()
        
        if client is None:
            raise RuntimeError("Weave 클라이언트를 초기화할 수 없습니다.")
        
        target_model = model_filter or self.model_name
        
        try:
            # Inspect AI 평가 trace 검색
            # op_name이 "inspect_ai" 또는 벤치마크 이름을 포함하는 것들
            calls = client.calls(
                filter={
                    "trace_roots_only": True,  # 최상위 trace만
                },
                limit=limit,
            )
            
            for call in calls:
                # 모델 필터링
                call_model = self._extract_model_from_call(call)
                if target_model and call_model and target_model not in call_model:
                    continue
                
                # 벤치마크 이름 추출
                benchmark_name = self._extract_benchmark_from_call(call)
                if not benchmark_name:
                    continue
                
                # 벤치마크 필터링
                if benchmark_filter and benchmark_name not in benchmark_filter:
                    continue
                
                # 점수 추출
                scores = self._extract_scores_from_call(call, benchmark_name)
                if scores:
                    self.benchmark_results[benchmark_name] = scores
                    print(f"  ✓ {benchmark_name}: {scores}")
        
        except Exception as e:
            print(f"⚠️ Weave trace 수집 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_model_from_call(self, call) -> str | None:
        """Call에서 모델 이름 추출"""
        try:
            if hasattr(call, "inputs") and call.inputs:
                return call.inputs.get("model")
            if hasattr(call, "attributes") and call.attributes:
                return call.attributes.get("model")
        except Exception:
            pass
        return None
    
    def _extract_benchmark_from_call(self, call) -> str | None:
        """Call에서 벤치마크 이름 추출"""
        try:
            # op_name에서 추출
            op_name = getattr(call, "op_name", "") or ""
            for benchmark in BENCHMARK_CONFIG.keys():
                if benchmark in op_name.lower():
                    return benchmark
            
            # attributes에서 추출
            if hasattr(call, "attributes") and call.attributes:
                task = call.attributes.get("task") or call.attributes.get("benchmark")
                if task and task in BENCHMARK_CONFIG:
                    return task
            
            # inputs에서 추출
            if hasattr(call, "inputs") and call.inputs:
                task = call.inputs.get("task") or call.inputs.get("benchmark")
                if task and task in BENCHMARK_CONFIG:
                    return task
        except Exception:
            pass
        return None
    
    def _extract_scores_from_call(self, call, benchmark_name: str) -> dict | None:
        """Call에서 점수 추출"""
        try:
            output = call.output if hasattr(call, "output") else None
            if not output:
                return None
            
            config = BENCHMARK_CONFIG.get(benchmark_name, {})
            score_key = config.get("score_key", "accuracy")
            
            # output이 dict인 경우
            if isinstance(output, dict):
                score = extract_score_from_results(output, score_key)
                if score is not None:
                    return {"score": score}
            
            # 직접 값인 경우
            if isinstance(output, (int, float)):
                return {"score": output}
        
        except Exception:
            pass
        return None
    
    def build_leaderboard_df(self) -> pd.DataFrame:
        """
        수집된 벤치마크 결과로 리더보드 DataFrame 생성
        
        Returns:
            리더보드 DataFrame (GLP/ALT 점수 포함)
        """
        if not self.benchmark_results:
            raise ValueError("수집된 벤치마크 결과가 없습니다.")
        
        # 초기 DataFrame 생성
        data = {"model_name": [self.model_name]}
        
        # 각 벤치마크 결과를 GLP/ALT 카테고리로 매핑
        for benchmark_name, scores in self.benchmark_results.items():
            config = BENCHMARK_CONFIG.get(benchmark_name, {})
            mapper = config.get("mapper", {})
            
            # 단일 score인 경우
            if "score" in scores and len(mapper) == 1:
                category = list(mapper.values())[0]
                if category not in data:
                    data[category] = []
                # 여러 벤치마크가 같은 카테고리에 매핑되면 평균 계산을 위해 리스트로 저장
                if len(data[category]) < 1:
                    data[category].append(scores["score"])
                else:
                    # 이미 값이 있으면 평균 계산
                    data[category][0] = (data[category][0] + scores["score"]) / 2
            
            # 여러 점수가 있는 경우 (mtbench 등)
            else:
                for score_field, category in mapper.items():
                    if score_field in scores:
                        if category not in data:
                            data[category] = []
                        if len(data[category]) < 1:
                            data[category].append(scores[score_field])
                        else:
                            data[category][0] = (data[category][0] + scores[score_field]) / 2
        
        # 모든 값이 같은 길이인지 확인
        max_len = max(len(v) if isinstance(v, list) else 1 for v in data.values())
        for key in data:
            if isinstance(data[key], list) and len(data[key]) < max_len:
                data[key].extend([float('nan')] * (max_len - len(data[key])))
        
        df = pd.DataFrame(data)
        
        # GLP/ALT 평균 계산
        df['범용언어성능(GLP)_AVG'] = weighted_average(df, GLP_COLUMN_WEIGHT)
        df['가치정렬성능(ALT)_AVG'] = weighted_average(df, ALT_COLUMN_WEIGHT)
        
        # 최종 점수 계산
        glp_score = df['범용언어성능(GLP)_AVG'].iloc[0] if '범용언어성능(GLP)_AVG' in df.columns else float('nan')
        alt_score = df['가치정렬성능(ALT)_AVG'].iloc[0] if '가치정렬성능(ALT)_AVG' in df.columns else float('nan')
        
        if pd.notna(glp_score) and pd.notna(alt_score):
            df['FINAL_SCORE'] = (glp_score + alt_score) / 2
        elif pd.notna(glp_score):
            df['FINAL_SCORE'] = glp_score
        elif pd.notna(alt_score):
            df['FINAL_SCORE'] = alt_score
        else:
            df['FINAL_SCORE'] = float('nan')
        
        # 메타데이터 추가
        df['release_date'] = pd.to_datetime(self.release_date, format='%Y-%m-%d', errors='coerce')
        df['size_category'] = self.size_category
        df['model_size'] = self.model_size
        
        # 컬럼 정렬
        desired_columns = [
            'model_name', 'release_date', 'size_category', 'model_size', 
            'FINAL_SCORE', '범용언어성능(GLP)_AVG', '가치정렬성능(ALT)_AVG'
        ] + list(GLP_COLUMN_WEIGHT.keys()) + list(ALT_COLUMN_WEIGHT.keys())
        
        existing_columns = [col for col in desired_columns if col in df.columns]
        return df[existing_columns]
    
    def build_radar_tables(
        self,
        df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        레이더 차트용 테이블 생성
        
        Args:
            df: 리더보드 DataFrame (없으면 새로 생성)
        
        Returns:
            (glp_radar_table, glp_detail_radar_table, 
             alt_radar_table, alt_detail_radar_table)
        """
        if df is None:
            df = self.build_leaderboard_df()
        
        # GLP 레이더 테이블
        glp_cols = [c for c in GLP_COLUMN_MAPPER.keys() if c in df.columns]
        if glp_cols:
            glp_radar_table = (
                df[glp_cols]
                .rename(columns=GLP_COLUMN_MAPPER)
                .transpose()
                .reset_index()
                .groupby("index")
                .mean()
                .reset_index()
                .rename(columns={'index': 'category', 0: 'score'})
            )
            glp_detail_radar_table = (
                df[glp_cols]
                .transpose()
                .reset_index()
                .rename(columns={'index': 'category', 0: 'score'})
            )
        else:
            glp_radar_table = pd.DataFrame(columns=['category', 'score'])
            glp_detail_radar_table = pd.DataFrame(columns=['category', 'score'])
        
        # ALT 레이더 테이블
        alt_cols = [c for c in ALT_COLUMN_MAPPER.keys() if c in df.columns]
        if alt_cols:
            alt_radar_table = (
                df[alt_cols]
                .rename(columns=ALT_COLUMN_MAPPER)
                .transpose()
                .reset_index()
                .groupby("index")
                .mean()
                .reset_index()
                .rename(columns={'index': 'category', 0: 'score'})
            )
            alt_detail_radar_table = (
                df[alt_cols]
                .transpose()
                .reset_index()
                .rename(columns={'index': 'category', 0: 'score'})
            )
        else:
            alt_radar_table = pd.DataFrame(columns=['category', 'score'])
            alt_detail_radar_table = pd.DataFrame(columns=['category', 'score'])
        
        return glp_radar_table, glp_detail_radar_table, alt_radar_table, alt_detail_radar_table
    
    def build_and_log(
        self,
        wandb_project: str | None = None,
        log_radar_tables: bool = True,
    ) -> pd.DataFrame:
        """
        리더보드 테이블 생성 및 W&B에 로깅
        
        Args:
            wandb_project: W&B 프로젝트 (없으면 self.project 사용)
            log_radar_tables: 레이더 테이블도 로깅할지 여부
        
        Returns:
            리더보드 DataFrame
        """
        if wandb is None:
            raise ImportError("wandb 패키지가 설치되어 있지 않습니다.")
        
        # 리더보드 테이블 생성
        leaderboard_df = self.build_leaderboard_df()
        
        # W&B 초기화 (필요한 경우)
        if self._wandb_run is None:
            self._wandb_run = wandb.init(
                project=wandb_project or self.project,
                entity=self.entity,
                job_type="leaderboard",
                name=f"leaderboard-{self.model_name}",
            )
        
        # 리더보드 테이블 로깅
        leaderboard_table = wandb.Table(dataframe=leaderboard_df)
        wandb.log({"leaderboard_table": leaderboard_table})
        
        # 레이더 테이블 로깅
        if log_radar_tables:
            glp_radar, glp_detail, alt_radar, alt_detail = self.build_radar_tables(leaderboard_df)
            
            wandb.log({
                "glp_radar_table": wandb.Table(dataframe=glp_radar),
                "glp_detail_radar_table": wandb.Table(dataframe=glp_detail),
                "alt_radar_table": wandb.Table(dataframe=alt_radar),
                "alt_detail_radar_table": wandb.Table(dataframe=alt_detail),
            })
        
        print(f"✅ 리더보드 테이블이 W&B에 로깅되었습니다.")
        print(f"   프로젝트: {self.entity}/{wandb_project or self.project}")
        
        return leaderboard_df
    
    def finish(self) -> None:
        """W&B run 종료"""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None


# =============================================================================
# 편의 함수
# =============================================================================

def create_leaderboard_from_benchmarks(
    entity: str,
    project: str,
    model_name: str,
    benchmark_results: dict[str, dict[str, float]],
    release_date: str = "unknown",
    size_category: str = "unknown",
    model_size: str = "unknown",
    log_to_wandb: bool = True,
) -> pd.DataFrame:
    """
    벤치마크 결과로 리더보드 테이블 생성 (편의 함수)
    
    Args:
        entity: W&B entity
        project: W&B 프로젝트
        model_name: 모델 이름
        benchmark_results: 벤치마크 결과 딕셔너리
            예: {"ko_hle": {"score": 0.85}, "kmmlu": {"score": 0.72}}
        release_date: 모델 출시일
        size_category: 모델 크기 카테고리
        model_size: 모델 크기
        log_to_wandb: W&B에 로깅할지 여부
    
    Returns:
        리더보드 DataFrame
    
    Example:
        >>> df = create_leaderboard_from_benchmarks(
        ...     entity="my-team",
        ...     project="korean-llm-eval",
        ...     model_name="gpt-4o",
        ...     benchmark_results={
        ...         "ko_hle": {"score": 0.42},
        ...         "kmmlu": {"score": 0.78},
        ...         "kmmlu_pro": {"score": 0.65},
        ...         "kobbq": {"score": 0.82},
        ...         "korean_hate_speech": {"score": 0.91},
        ...     },
        ...     release_date="2024-05-13",
        ...     size_category="flagship",
        ... )
    """
    builder = LeaderboardTableBuilder(
        entity=entity,
        project=project,
        model_name=model_name,
        release_date=release_date,
        size_category=size_category,
        model_size=model_size,
    )
    
    for benchmark_name, scores in benchmark_results.items():
        builder.add_benchmark_result(benchmark_name, scores)
    
    if log_to_wandb:
        return builder.build_and_log()
    else:
        return builder.build_leaderboard_df()


def aggregate_multiple_models(
    model_results: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    여러 모델의 리더보드 테이블 통합
    
    Args:
        model_results: 각 모델의 리더보드 DataFrame 리스트
    
    Returns:
        통합된 리더보드 DataFrame
    """
    if not model_results:
        return pd.DataFrame()
    
    return pd.concat(model_results, ignore_index=True)


# =============================================================================
# CLI 지원
# =============================================================================

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="평가 결과로 리더보드 테이블 생성"
    )
    parser.add_argument(
        "--entity", "-e",
        required=True,
        help="W&B entity (팀 또는 사용자 이름)"
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="W&B 프로젝트 이름"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="모델 이름"
    )
    parser.add_argument(
        "--release-date",
        default="unknown",
        help="모델 출시일 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--size-category",
        default="unknown",
        help="모델 크기 카테고리"
    )
    parser.add_argument(
        "--model-size",
        default="unknown",
        help="모델 파라미터 수"
    )
    parser.add_argument(
        "--from-weave",
        action="store_true",
        help="Weave trace에서 결과 수집"
    )
    parser.add_argument(
        "--output", "-o",
        help="결과를 저장할 CSV 파일 경로"
    )
    
    args = parser.parse_args()
    
    builder = LeaderboardTableBuilder(
        entity=args.entity,
        project=args.project,
        model_name=args.model,
        release_date=args.release_date,
        size_category=args.size_category,
        model_size=args.model_size,
    )
    
    if args.from_weave:
        print(f"🔍 Weave trace에서 결과 수집 중...")
        builder.collect_from_weave_traces()
    
    if builder.benchmark_results:
        df = builder.build_and_log()
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"📁 결과가 {args.output}에 저장되었습니다.")
        
        print("\n📊 리더보드 테이블:")
        print(df.to_string())
    else:
        print("❌ 수집된 벤치마크 결과가 없습니다.")


if __name__ == "__main__":
    main()


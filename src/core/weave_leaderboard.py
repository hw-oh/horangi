"""
Weave Leaderboard 자동 생성 모듈

Inspect AI 평가 결과에서 Weave Leaderboard를 자동으로 생성합니다.
이 리더보드는 Weave UI에서 모델 간 성능 비교를 가능하게 합니다.

사용법:
    # run_eval.py에서 자동 호출됨
    from core.weave_leaderboard import create_weave_leaderboard
    
    create_weave_leaderboard(
        entity="wandb-korea",
        project="korean-llm-eval",
        model_name="gpt-4o",
    )

Note:
    - W&B Models 테이블 (leaderboard_table.py)과는 별개로 작동합니다.
    - Weave UI의 Leaderboard 기능을 사용합니다.
"""

from __future__ import annotations

import weave
from weave.flow import leaderboard
from weave.trace import urls as weave_urls


# 리더보드 설정
LEADERBOARD_REF = "Korean-LLM-Leaderboard"
LEADERBOARD_NAME = "Korean LLM Leaderboard"
LEADERBOARD_DESCRIPTION = """한국어 LLM 벤치마크 모델 성능 비교 리더보드

이 리더보드는 Inspect AI 평가 결과에서 자동으로 생성되었습니다.
다양한 벤치마크에서 모델들의 성능을 비교해볼 수 있습니다.

📊 벤치마크 카테고리:
- 언어 이해: ko_hellaswag, kmmlu, kmmlu_pro, haerae_bench
- 추론: ko_aime2025, ko_gsm8k, ko_arc_agi
- 지시 따르기: ifeval_ko, ko_balt_700
- 안전성/윤리: ko_moral, kobbq, korean_hate_speech
- 환각: ko_hallulens (wikiqa, longwiki, nonexistent)
- 지식: ko_truthful_qa, ko_hle
- 도구 사용: bfcl
- 대화: mtbench_ko
- 코딩: swebench_verified_official_80
"""


def get_evaluation_ref(entity: str, project: str, benchmark: str) -> str | None:
    """
    벤치마크에 해당하는 evaluation 객체의 실제 ref를 가져옵니다.
    
    :latest 태그는 Leaderboard에서 작동하지 않으므로,
    실제 digest가 포함된 ref를 반환합니다.
    """
    from weave.trace.ref_util import get_ref
    
    try:
        eval_name = f"{benchmark}-evaluation"
        eval_obj = weave.ref(f"{eval_name}:latest").get()
        ref = get_ref(eval_obj)
        if ref:
            return ref.uri()
    except Exception:
        pass
    
    return None


def build_columns_from_benchmarks(
    benchmarks: list[str],
    entity: str,
    project: str,
) -> list[leaderboard.LeaderboardColumn]:
    """
    벤치마크 이름 목록에서 LeaderboardColumn 생성
    
    각 벤치마크의 evaluation ref를 동적으로 가져와 컬럼을 생성합니다.
    
    Args:
        benchmarks: 벤치마크 이름 리스트
        entity: Weave entity
        project: Weave 프로젝트 이름
    
    Returns:
        LeaderboardColumn 리스트
    """
    # 벤치마크별 주요 메트릭 매핑
    # (scorer_name, summary_metric_path) 형태
    # output 구조: {"scorer_name": {"metric": value, ...}, ...}
    BENCHMARK_METRICS = {
        # 기본 choice scorer
        "ko_hellaswag": ("choice", "true_fraction"),
        "ko_balt_700_syntax": ("choice", "true_fraction"),
        "ko_balt_700_semantic": ("choice", "true_fraction"),
        "haerae_bench_v1_rc": ("choice", "true_fraction"),
        "haerae_bench_v1_wo_rc": ("choice", "true_fraction"),
        "kmmlu": ("choice", "true_fraction"),
        "kmmlu_pro": ("choice", "true_fraction"),
        "ko_truthful_qa": ("choice", "true_fraction"),
        "ko_moral": ("choice", "true_fraction"),
        "korean_hate_speech": ("choice", "true_fraction"),
        
        # model_graded_qa scorer
        "ko_aime2025": ("model_graded_qa", "true_fraction"),
        "ko_gsm8k": ("model_graded_qa", "true_fraction"),
        
        # 특수 scorer
        "ifeval_ko": ("instruction_following", "prompt_level_strict.true_fraction"),
        "ko_arc_agi": ("grid_match", "true_fraction"),
        "squad_kor_v1": ("f1", "mean"),
        
        # KoBBQ
        "kobbq": ("kobbq_scorer", "true_fraction"),
        
        # HLE
        "ko_hle": ("hle_grader", "true_fraction"),
        
        # HalluLens
        "ko_hallulens_wikiqa": ("hallulens_qa", "true_fraction"),
        "ko_hallulens_longwiki": ("hallulens_qa", "true_fraction"),
        "ko_hallulens_nonexistent": ("hallulens_refusal", "true_fraction"),
        
        # BFCL
        "bfcl": ("bfcl_scorer", "true_fraction"),
        
        # MT-Bench
        "mtbench_ko": ("mtbench_scorer", "mean"),
        
        # SWE-bench
        "swebench_verified_official_80": ("swebench_server_scorer", "true_fraction"),
    }
    
    columns = []
    
    for benchmark in benchmarks:
        # 실제 evaluation ref 가져오기 (digest 포함)
        eval_ref = get_evaluation_ref(entity, project, benchmark)
        
        if not eval_ref:
            print(f"   ⚠️ {benchmark}-evaluation 객체를 찾을 수 없음")
            continue
        
        # 해당 벤치마크의 메트릭 가져오기
        scorer_name, metric_path = BENCHMARK_METRICS.get(
            benchmark, ("output", "true_fraction")
        )
        
        columns.append(
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref,
                scorer_name=scorer_name,
                summary_metric_path=metric_path,
                should_minimize=False,
            )
        )
        print(f"   ✓ {benchmark}: {scorer_name}.{metric_path}")
    
    return columns


def create_weave_leaderboard(
    entity: str,
    project: str,
    benchmarks: list[str] | None = None,
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
) -> str | None:
    """
    Weave Leaderboard 생성/업데이트
    
    벤치마크 목록을 받아서 Weave Leaderboard를 생성합니다.
    기존 리더보드가 있으면 새 컬럼을 병합합니다.
    
    Args:
        entity: Weave entity (팀 또는 사용자 이름)
        project: Weave 프로젝트 이름
        benchmarks: 벤치마크 이름 리스트 (없으면 기본 목록 사용)
        name: 리더보드 이름
        description: 리더보드 설명
    
    Returns:
        리더보드 URL (성공 시) 또는 None (실패 시)
    """
    print(f"\n{'='*60}")
    print(f"🏆 Weave Leaderboard 생성")
    print(f"{'='*60}")
    
    # 기본 벤치마크 목록
    DEFAULT_BENCHMARKS = [
        "ko_hellaswag",
        "ko_aime2025",
        "ifeval_ko",
        "ko_balt_700_syntax",
        "ko_balt_700_semantic",
        "haerae_bench_v1_rc",
        "haerae_bench_v1_wo_rc",
        "kmmlu",
        "kmmlu_pro",
        "squad_kor_v1",
        "ko_truthful_qa",
        "ko_moral",
        "ko_arc_agi",
        "ko_gsm8k",
        "korean_hate_speech",
        "kobbq",
        "ko_hle",
        "ko_hallulens_wikiqa",
        "ko_hallulens_longwiki",
        "ko_hallulens_nonexistent",
        "bfcl",
        "mtbench_ko",
        "swebench_verified_official_80",
    ]
    
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    
    # Weave 초기화
    client = weave.get_client()
    if client is None:
        weave.init(f"{entity}/{project}")
        client = weave.get_client()
    
    if client is None:
        print("❌ Weave 클라이언트 초기화 실패")
        return None
    
    try:
        # 1. LeaderboardColumn 생성
        print(f"📊 {len(benchmarks)}개 벤치마크에서 LeaderboardColumn 생성 중...")
        new_columns = build_columns_from_benchmarks(benchmarks, entity, project)
        
        if not new_columns:
            print("⚠️ 생성할 컬럼이 없습니다.")
            return None
        
        print(f"   새 컬럼: {len(new_columns)}개")
        
        # 3. 기존 리더보드 가져오기 (있다면)
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
                print(f"   기존 컬럼: {len(existing_columns)}개")
        except Exception:
            print("   기존 리더보드 없음 - 새로 생성")
        
        # 4. 컬럼 병합 (중복 제거)
        merged_columns = list(
            {
                (
                    column.evaluation_object_ref,
                    column.scorer_name,
                    column.summary_metric_path,
                    column.should_minimize,
                ): column
                for column in (existing_columns or []) + new_columns
            }.values()
        )
        
        print(f"\n📈 총 {len(merged_columns)}개 컬럼으로 리더보드 생성")
        
        # 5. 리더보드 생성 및 발행
        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=merged_columns,
        )
        ref = weave.publish(spec, name=LEADERBOARD_REF)
        
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )
        
        print(f"\n✅ Weave Leaderboard 생성 완료!")
        print(f"🔗 URL: {url}")
        
        return url
        
    except Exception as e:
        print(f"❌ Leaderboard 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_weave_leaderboard_from_active_loggers(
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
) -> str | None:
    """
    활성화된 EvaluationLogger에서 Weave Leaderboard 생성
    
    이 함수는 같은 프로세스 내에서 evaluation이 실행된 경우에만 작동합니다.
    subprocess로 실행된 경우에는 create_weave_leaderboard()를 사용하세요.
    
    Args:
        name: 리더보드 이름
        description: 리더보드 설명
    
    Returns:
        리더보드 URL (성공 시) 또는 None (실패 시)
    """
    from weave.evaluation.eval_imperative import _active_evaluation_loggers
    from weave.trace.ref_util import get_ref
    
    client = weave.get_client()
    if client is None:
        print("❌ Weave 클라이언트가 초기화되지 않았습니다.")
        return None
    
    try:
        # 활성 logger에서 컬럼 빌드
        new_columns: list[leaderboard.LeaderboardColumn] = []
        
        for eval_logger in _active_evaluation_loggers:
            eval_output = eval_logger._evaluate_call and (eval_logger._evaluate_call.output or {})
            output_scorer = eval_output.get("output", {})
            
            for metric_name, metric_values in output_scorer.items():
                if not isinstance(metric_values, dict):
                    continue
                    
                for m_value in metric_values.keys():
                    if "err" in m_value.lower():
                        continue
                    
                    new_columns.append(
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=get_ref(
                                eval_logger._pseudo_evaluation
                            ).uri(),
                            scorer_name="output",
                            summary_metric_path=f"{metric_name}.{m_value}",
                            should_minimize=False,
                        )
                    )
        
        if not new_columns:
            print("⚠️ 활성 evaluation logger가 없습니다.")
            return None
        
        # 기존 리더보드와 병합
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
        except Exception:
            pass
        
        merged_columns = list(
            {
                (
                    column.evaluation_object_ref,
                    column.scorer_name,
                    column.summary_metric_path,
                    column.should_minimize,
                ): column
                for column in (existing_columns or []) + new_columns
            }.values()
        )
        
        # 리더보드 발행
        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=merged_columns,
        )
        ref = weave.publish(spec, name=LEADERBOARD_REF)
        
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )
        
        print(f"✅ Weave Leaderboard 생성 완료!")
        print(f"🔗 URL: {url}")
        
        return url
        
    except Exception as e:
        print(f"❌ Leaderboard 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Weave Leaderboard 생성")
    parser.add_argument("--entity", "-e", required=True, help="Weave entity")
    parser.add_argument("--project", "-p", required=True, help="Weave project")
    parser.add_argument("--benchmarks", "-b", nargs="+", help="벤치마크 목록 (기본: 전체)")
    parser.add_argument("--name", default=LEADERBOARD_NAME, help="리더보드 이름")
    
    args = parser.parse_args()
    
    create_weave_leaderboard(
        entity=args.entity,
        project=args.project,
        benchmarks=args.benchmarks,
        name=args.name,
    )


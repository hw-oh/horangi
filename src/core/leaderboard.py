"""
평가 결과를 Weave 리더보드로 생성하는 모듈

이 모듈은 Inspect AI 평가 결과를 Weave 리더보드로 만들어서
모델 간 성능 비교를 가능하게 합니다.

사용법:
    # 평가 실행 후 CLI에서 리더보드 생성
    uv run horangi leaderboard --project <entity>/<project>
    
    # 또는 코드에서 직접 사용
    from core.leaderboard import create_leaderboard
    create_leaderboard()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import wandb
import weave
from weave import weave_urls
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

if TYPE_CHECKING:
    from weave.evaluation.eval_imperative import EvaluationLogger

# 리더보드 설정
LEADERBOARD_REF = "Inspect-AI-Leaderboard"
LEADERBOARD_NAME = "Inspect AI Leaderboard"
LEADERBOARD_DESCRIPTION = """한국어 LLM 벤치마크 모델 성능 비교 리더보드

이 리더보드는 Inspect AI 평가 결과에서 자동으로 생성되었습니다.
다양한 벤치마크에서 모델들의 성능을 비교해볼 수 있습니다.

벤치마크 목록:
- ko_hellaswag, ko_aime2025, ifeval_ko, ko_balt_700
- haerae_bench_v1, kmmlu, kmmlu_pro, ko_truthful_qa
- ko_moral, ko_arc_agi, ko_gsm8k
- korean_hate_speech, kobbq, ko_hle
- ko_hallulens (wikiqa, longwiki, generated, mixed, nonexistent)
- bfcl (Function Calling)
- mtbench_ko (Multi-turn)
- swebench_verified_official_80 (Coding)
"""


def build_columns_from_eval_logger(
    eval_logger: "EvaluationLogger",
) -> list[leaderboard.LeaderboardColumn]:
    """
    단일 EvaluationLogger에서 리더보드 컬럼 생성
    
    평가 결과의 'output' scorer에서 메트릭을 추출하여
    LeaderboardColumn 객체 리스트를 생성합니다.
    
    Args:
        eval_logger: Weave EvaluationLogger 인스턴스
    
    Returns:
        LeaderboardColumn 객체 리스트
    """
    eval_output = eval_logger._evaluate_call and (eval_logger._evaluate_call.output or {})
    output_scorer = eval_output.get("output", {})
    lb_columns = []
    
    for metric_name, metric_values in output_scorer.items():
        for m_value in metric_values:
            # 에러 메트릭 스킵
            if "err" in m_value:
                continue

            lb_columns.append(
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=get_ref(
                        eval_logger._pseudo_evaluation
                    ).uri(),
                    scorer_name="output",
                    summary_metric_path=f"{metric_name}.{m_value}",
                    should_minimize=False,
                )
            )

    return lb_columns


def build_columns_from_evaluation_ref(
    evaluation_ref: str,
    scorer_name: str = "output",
) -> list[leaderboard.LeaderboardColumn]:
    """
    Weave 평가 참조에서 리더보드 컬럼 생성
    
    기존에 저장된 평가 결과 ref를 사용하여 컬럼을 생성합니다.
    
    Args:
        evaluation_ref: Weave 평가 객체 참조 URI
        scorer_name: Scorer 이름 (기본: "output")
    
    Returns:
        LeaderboardColumn 객체 리스트
    """
    try:
        evaluation = weave.ref(evaluation_ref).get()
    except Exception as e:
        print(f"⚠️ 평가를 가져올 수 없습니다: {evaluation_ref} ({e})")
        return []
    
    lb_columns = []
    
    # 평가 결과에서 메트릭 추출
    # Weave evaluation 객체의 구조에 따라 메트릭을 추출
    if hasattr(evaluation, "summary") and evaluation.summary:
        summary = evaluation.summary
        for metric_name, metric_values in summary.items():
            if isinstance(metric_values, dict):
                for m_value in metric_values.keys():
                    if "err" in m_value:
                        continue
                    lb_columns.append(
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=evaluation_ref,
                            scorer_name=scorer_name,
                            summary_metric_path=f"{metric_name}.{m_value}",
                            should_minimize=False,
                        )
                    )
            else:
                # 단일 값인 경우
                lb_columns.append(
                    leaderboard.LeaderboardColumn(
                        evaluation_object_ref=evaluation_ref,
                        scorer_name=scorer_name,
                        summary_metric_path=metric_name,
                        should_minimize=False,
                    )
                )
    
    return lb_columns


def get_evaluations_from_project(
    entity: str,
    project: str,
    limit: int = 100,
) -> list[str]:
    """
    프로젝트에서 평가 객체 참조 목록 가져오기
    
    Args:
        entity: Weave entity (팀 또는 사용자 이름)
        project: Weave 프로젝트 이름
        limit: 가져올 최대 평가 수
    
    Returns:
        평가 객체 참조 URI 리스트
    """
    client = weave.get_client()
    if client is None:
        weave.init(f"{entity}/{project}")
        client = weave.get_client()
    
    evaluation_refs = []
    
    try:
        # Weave API를 통해 평가 객체들을 검색
        # Note: 실제 구현은 Weave API 구조에 따라 다를 수 있음
        calls = client.calls(
            filter={
                "op_name": {"$regex": "Evaluation.evaluate"},
            },
            limit=limit,
        )
        
        for call in calls:
            if call.output and hasattr(call, "ref"):
                evaluation_refs.append(call.ref.uri())
    except Exception as e:
        print(f"⚠️ 평가 목록을 가져오는 중 오류 발생: {e}")
    
    return evaluation_refs


def create_leaderboard(
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
    entity: str | None = None,
    project: str | None = None,
) -> str | None:
    """
    평가 결과에서 Weave 리더보드 생성
    
    활성화된 EvaluationLogger들에서 메트릭을 추출하거나,
    entity/project가 지정된 경우 해당 프로젝트의 평가들을 사용합니다.
    
    기존 리더보드가 있으면 새 컬럼을 병합합니다.
    
    Args:
        name: 리더보드 이름
        description: 리더보드 설명
        entity: Weave entity (팀 또는 사용자 이름)
        project: Weave 프로젝트 이름
    
    Returns:
        리더보드 URL (성공 시) 또는 None (실패 시)
    """
    client = weave.get_client()
    
    if client is None:
        if entity and project:
            weave.init(f"{entity}/{project}")
            client = weave.get_client()
        else:
            print("❌ Weave 클라이언트가 초기화되지 않았습니다.")
            print("   entity와 project를 지정하거나, 먼저 weave.init()을 호출하세요.")
            return None

    try:
        # 새 컬럼 빌드
        new_columns: list[leaderboard.LeaderboardColumn] = []
        
        # 1. 활성화된 evaluation logger들에서 컬럼 추출
        try:
            from weave.evaluation.eval_imperative import _active_evaluation_loggers
            for eval_logger in _active_evaluation_loggers:
                new_columns.extend(build_columns_from_eval_logger(eval_logger))
        except ImportError:
            pass
        
        # 2. entity/project가 지정된 경우 프로젝트에서 평가 검색
        if entity and project and not new_columns:
            print(f"🔍 프로젝트 {entity}/{project}에서 평가 검색 중...")
            eval_refs = get_evaluations_from_project(entity, project)
            for eval_ref in eval_refs:
                new_columns.extend(build_columns_from_evaluation_ref(eval_ref))

        if not new_columns:
            print("⚠️ 평가 결과가 없습니다. 먼저 평가를 실행하세요.")
            return None

        # 기존 리더보드 가져오기 (있다면)
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
                print(f"📊 기존 리더보드에서 {len(existing_columns)}개 컬럼 발견")
        except Exception:
            # 기존 리더보드 없음 - 새로 생성
            print("📝 새 리더보드 생성")
            existing_columns = []

        # 컬럼 병합 (중복 제거)
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

        print(f"📈 총 {len(merged_columns)}개 컬럼으로 리더보드 생성")

        # 리더보드 생성 및 발행
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

        print(f"✅ 리더보드가 생성되었습니다!")
        print(f"🔗 Weave에서 보기: {url}")
        
        return url

    except Exception as e:
        wandb.termerror(f"리더보드 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_leaderboard_from_weave_traces(
    entity: str,
    project: str,
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
    trace_filter: dict | None = None,
) -> str | None:
    """
    Weave 트레이스에서 평가 결과를 가져와 리더보드 업데이트
    
    이 함수는 Weave 프로젝트의 트레이스를 검색하여
    평가 결과를 추출하고 리더보드를 업데이트합니다.
    
    Args:
        entity: Weave entity (팀 또는 사용자 이름)
        project: Weave 프로젝트 이름
        name: 리더보드 이름
        description: 리더보드 설명
        trace_filter: 트레이스 필터 (op_name, status 등)
    
    Returns:
        리더보드 URL (성공 시) 또는 None (실패 시)
    """
    # Weave 초기화
    weave.init(f"{entity}/{project}")
    
    return create_leaderboard(
        name=name,
        description=description,
        entity=entity,
        project=project,
    )


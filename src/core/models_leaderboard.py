"""
W&B 리더보드 테이블 생성 모듈

평가 결과를 W&B Table로 변환하여 로깅합니다.
README_ko.md의 택소노미를 기반으로 GLP/ALT 점수를 계산합니다.
"""

from typing import Any

import pandas as pd
import wandb


# 벤치마크 → 택소노미 매핑
BENCHMARK_TAXONOMY = {
    # GLP - 구문해석
    "ko_balt_700_syntax": {
        "category": "GLP_구문해석",
        "score_key": "score",
    },
    # GLP - 의미해석
    "ko_balt_700_semantic": {
        "category": "GLP_의미해석",
        "score_key": "score",
    },
    "haerae_bench_v1_rc": {
        "category": "GLP_의미해석",
        "score_key": "score",
    },
    # GLP - 표현
    "ko_mtbench": {
        "category": "GLP_표현",
        "score_key": "score",
        "sub_scores": {
            # 실제 메트릭 이름: mtbench_scorer_XXX_score
            "mtbench_scorer_roleplay_score": "GLP_표현",
            "mtbench_scorer_humanities_score": "GLP_표현",
            "mtbench_scorer_writing_score": "GLP_표현",
            "mtbench_scorer_reasoning_score": "GLP_논리추론",
            "mtbench_scorer_coding_score": "GLP_코딩",
            "mtbench_scorer_math_score": "GLP_수학추론",
            "mtbench_scorer_stem_score": "GLP_전문지식",
            "mtbench_scorer_extraction_score": "GLP_정보검색",
        },
    },
    # GLP - 정보검색
    "squad_kor_v1": {
        "category": "GLP_정보검색",
        "score_key": "score",
    },
    # GLP - 일반지식
    "kmmlu": {
        "category": "GLP_일반지식",
        "score_key": "score",
    },
    "haerae_bench_v1_wo_rc": {
        "category": "GLP_일반지식",
        "score_key": "score",
    },
    # GLP - 전문지식
    "kmmlu_pro": {
        "category": "GLP_전문지식",
        "score_key": "score",
    },
    "ko_hle": {
        "category": "GLP_전문지식",
        "score_key": "score",
    },
    # GLP - 상식추론
    "ko_hellaswag": {
        "category": "GLP_상식추론",
        "score_key": "score",
    },
    # GLP - 수학추론
    "ko_gsm8k": {
        "category": "GLP_수학추론",
        "score_key": "score",
    },
    "ko_aime2025": {
        "category": "GLP_수학추론",
        "score_key": "score",
    },
    # GLP - 추상추론
    "ko_arc_agi": {
        "category": "GLP_추상추론",
        "score_key": "score",
    },
    # GLP - 코딩
    "swebench_verified_official_80": {
        "category": "GLP_코딩",
        "score_key": "score",
    },
    # GLP - 함수호출
    "bfcl": {
        "category": "GLP_함수호출",
        "score_key": "score",
    },
    # ALT - 제어성
    "ifeval_ko": {
        "category": "ALT_제어성",
        "score_key": "score",
    },
    # ALT - 윤리/도덕
    "ko_moral": {
        "category": "ALT_윤리도덕",
        "score_key": "score",
    },
    # ALT - 유해성방지
    "korean_hate_speech": {
        "category": "ALT_유해성방지",
        "score_key": "score",
    },
    # ALT - 편향성방지
    "kobbq": {
        "category": "ALT_편향성방지",
        "score_key": "score",
    },
    # ALT - 환각방지
    "ko_truthful_qa": {
        "category": "ALT_환각방지",
        "score_key": "score",
    },
    "ko_hallulens_wikiqa": {
        "category": "ALT_환각방지",
        "score_key": "score",
    },
    "ko_hallulens_longwiki": {
        "category": "ALT_환각방지",
        "score_key": "score",
    },
    "ko_hallulens_nonexistent": {
        "category": "ALT_환각방지",
        "score_key": "score",
    },
}

# GLP 카테고리별 가중치
GLP_COLUMN_WEIGHT = {
    "GLP_구문해석": 1,
    "GLP_의미해석": 1,
    "GLP_표현": 1,
    "GLP_정보검색": 1,
    "GLP_일반지식": 2,
    "GLP_전문지식": 2,
    "GLP_상식추론": 1,
    "GLP_수학추론": 2,
    "GLP_논리추론": 2,
    "GLP_추상추론": 2,
    "GLP_함수호출": 2,
    "GLP_코딩": 2,
}

# ALT 카테고리별 가중치
ALT_COLUMN_WEIGHT = {
    "ALT_제어성": 1,
    "ALT_유해성방지": 1,
    "ALT_편향성방지": 1,
    "ALT_윤리도덕": 1,
    "ALT_환각방지": 1,
}

# GLP 세부 카테고리 → 상위 카테고리 매핑 (레이더 차트용)
GLP_CATEGORY_MAPPER = {
    "GLP_구문해석": "기본언어성능",
    "GLP_의미해석": "기본언어성능",
    "GLP_표현": "응용언어성능",
    "GLP_정보검색": "응용언어성능",
    "GLP_일반지식": "지식/질의응답",
    "GLP_전문지식": "지식/질의응답",
    "GLP_상식추론": "추론능력",
    "GLP_수학추론": "추론능력",
    "GLP_논리추론": "추론능력",
    "GLP_추상추론": "추론능력",
    "GLP_함수호출": "어플리케이션개발",
    "GLP_코딩": "어플리케이션개발",
}

# ALT 세부 카테고리 → 상위 카테고리 매핑 (레이더 차트용)
ALT_CATEGORY_MAPPER = {
    "ALT_제어성": "제어성",
    "ALT_유해성방지": "유해성방지",
    "ALT_편향성방지": "편향성방지",
    "ALT_윤리도덕": "윤리/도덕",
    "ALT_환각방지": "환각방지",
}


def weighted_average(df: pd.DataFrame, weights_dict: dict[str, float]) -> pd.Series:
    """가중 평균 계산"""
    cols = [c for c in weights_dict.keys() if c in df.columns]
    if not cols:
        return pd.Series([None] * len(df), index=df.index)
    weights = [weights_dict[c] for c in cols]
    return (df[cols].mul(weights, axis=1).sum(axis=1)) / sum(weights)


def create_leaderboard_table(
    model_name: str,
    benchmark_scores: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    벤치마크 점수를 리더보드 테이블로 변환
    
    Args:
        model_name: 모델 이름
        benchmark_scores: 벤치마크별 점수 딕셔너리
            예: {"kmmlu": {"score": 0.85, "details": {...}}, ...}
        metadata: 모델 메타데이터 (release_date, size_category, model_size 등)
    
    Returns:
        리더보드 DataFrame
    """
    if metadata is None:
        metadata = {}
    
    # 카테고리별 점수 수집
    category_scores: dict[str, list[float]] = {}
    
    # 디버깅: benchmark_scores 내용 출력
    print(f"   📊 Processing {len(benchmark_scores)} benchmarks...")
    
    for benchmark_name, score_info in benchmark_scores.items():
        main_score = score_info.get("score")
        
        if benchmark_name not in BENCHMARK_TAXONOMY:
            print(f"   ⚠️ Benchmark '{benchmark_name}' not in BENCHMARK_TAXONOMY (score: {main_score})")
            continue
        
        taxonomy = BENCHMARK_TAXONOMY[benchmark_name]
        main_category = taxonomy["category"]
        
        if main_score is not None:
            if main_category not in category_scores:
                category_scores[main_category] = []
            category_scores[main_category].append(main_score)
            print(f"   ✅ {benchmark_name} → {main_category}: {main_score}")
        else:
            print(f"   ⚠️ {benchmark_name}: score is None")
        
        # sub_scores가 있는 경우 (ko_mtbench 등)
        if "sub_scores" in taxonomy and score_info.get("details"):
            for detail_key, sub_category in taxonomy["sub_scores"].items():
                detail_value = score_info["details"].get(detail_key)
                if detail_value is not None:
                    if sub_category not in category_scores:
                        category_scores[sub_category] = []
                    category_scores[sub_category].append(detail_value)
                    print(f"   ✅ {benchmark_name}.{detail_key} → {sub_category}: {detail_value}")
    
    # 카테고리별 평균 계산
    category_means = {}
    for category, scores in category_scores.items():
        if scores:
            category_means[category] = sum(scores) / len(scores)
    
    print(f"   📈 Category scores: {len(category_means)} categories")
    for cat, val in category_means.items():
        print(f"      {cat}: {val:.4f}")
    
    # DataFrame 생성
    row_data = {"model_name": model_name}
    row_data.update(category_means)
    
    df = pd.DataFrame([row_data])
    df.set_index("model_name", inplace=True)
    
    # GLP/ALT 평균 계산
    df["범용언어성능(GLP)_AVG"] = weighted_average(df, GLP_COLUMN_WEIGHT)
    df["가치정렬성능(ALT)_AVG"] = weighted_average(df, ALT_COLUMN_WEIGHT)
    
    # 최종 점수 계산
    glp_avg = df["범용언어성능(GLP)_AVG"].iloc[0]
    alt_avg = df["가치정렬성능(ALT)_AVG"].iloc[0]
    
    if pd.notna(glp_avg) and pd.notna(alt_avg):
        df["FINAL_SCORE"] = (glp_avg + alt_avg) / 2
    elif pd.notna(glp_avg):
        df["FINAL_SCORE"] = glp_avg
    elif pd.notna(alt_avg):
        df["FINAL_SCORE"] = alt_avg
    else:
        df["FINAL_SCORE"] = None
    
    # 메타데이터 추가
    df["release_date"] = metadata.get("release_date", "unknown")
    df["size_category"] = metadata.get("size_category", "unknown")
    df["model_size"] = metadata.get("model_size", "unknown")
    
    df = df.reset_index()
    
    # 컬럼 순서 정렬
    priority_columns = [
        "model_name",
        "release_date",
        "size_category",
        "model_size",
        "FINAL_SCORE",
        "범용언어성능(GLP)_AVG",
        "가치정렬성능(ALT)_AVG",
    ]
    glp_columns = [c for c in GLP_COLUMN_WEIGHT.keys() if c in df.columns]
    alt_columns = [c for c in ALT_COLUMN_WEIGHT.keys() if c in df.columns]
    
    ordered_columns = []
    for col in priority_columns + glp_columns + alt_columns:
        if col in df.columns and col not in ordered_columns:
            ordered_columns.append(col)
    
    # 나머지 컬럼 추가
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    
    return df[ordered_columns]


def create_radar_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    레이더 차트용 테이블 생성
    
    Args:
        df: 리더보드 DataFrame
    
    Returns:
        (glp_radar_table, glp_detail_radar_table, alt_radar_table, alt_detail_radar_table)
    """
    # GLP 레이더 테이블
    glp_cols = [c for c in GLP_CATEGORY_MAPPER.keys() if c in df.columns]
    if glp_cols:
        glp_detail_radar = df[glp_cols].T.reset_index()
        glp_detail_radar.columns = ["category", "score"]
        
        # 상위 카테고리로 그룹핑
        glp_detail_radar["group"] = glp_detail_radar["category"].map(GLP_CATEGORY_MAPPER)
        glp_radar = glp_detail_radar.groupby("group")["score"].mean().reset_index()
        glp_radar.columns = ["category", "score"]
    else:
        glp_radar = pd.DataFrame(columns=["category", "score"])
        glp_detail_radar = pd.DataFrame(columns=["category", "score"])
    
    # ALT 레이더 테이블
    alt_cols = [c for c in ALT_CATEGORY_MAPPER.keys() if c in df.columns]
    if alt_cols:
        alt_detail_radar = df[alt_cols].T.reset_index()
        alt_detail_radar.columns = ["category", "score"]
        
        # 상위 카테고리로 그룹핑
        alt_detail_radar["group"] = alt_detail_radar["category"].map(ALT_CATEGORY_MAPPER)
        alt_radar = alt_detail_radar.groupby("group")["score"].mean().reset_index()
        alt_radar.columns = ["category", "score"]
    else:
        alt_radar = pd.DataFrame(columns=["category", "score"])
        alt_detail_radar = pd.DataFrame(columns=["category", "score"])
    
    # 불필요한 컬럼 제거
    if "group" in glp_detail_radar.columns:
        glp_detail_radar = glp_detail_radar.drop(columns=["group"])
    if "group" in alt_detail_radar.columns:
        alt_detail_radar = alt_detail_radar.drop(columns=["group"])
    
    return glp_radar, glp_detail_radar, alt_radar, alt_detail_radar


def create_benchmark_detail_table(
    model_name: str,
    benchmark_scores: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    벤치마크별 상세 점수 테이블 생성
    
    Args:
        model_name: 모델 이름
        benchmark_scores: 벤치마크별 점수 딕셔너리
    
    Returns:
        벤치마크별 상세 DataFrame
    """
    rows = []
    
    for benchmark_name, score_info in benchmark_scores.items():
        row = {
            "model_name": model_name,
            "benchmark": benchmark_name,
            "score": score_info.get("score"),
        }
        
        # 택소노미 카테고리 추가
        if benchmark_name in BENCHMARK_TAXONOMY:
            row["category"] = BENCHMARK_TAXONOMY[benchmark_name]["category"]
        else:
            row["category"] = "기타"
        
        # 상세 점수 추가
        details = score_info.get("details", {})
        for key, value in details.items():
            row[f"detail_{key}"] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def log_leaderboard_tables(
    wandb_run: wandb.sdk.wandb_run.Run,
    model_name: str,
    benchmark_scores: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    W&B에 리더보드 테이블 로깅
    
    Args:
        wandb_run: W&B run 객체
        model_name: 모델 이름
        benchmark_scores: 벤치마크별 점수 딕셔너리
        metadata: 모델 메타데이터
    """
    if not benchmark_scores:
        print("⚠️ No benchmark scores to log")
        return
    
    # 리더보드 테이블 생성
    leaderboard_df = create_leaderboard_table(model_name, benchmark_scores, metadata)
    
    # 레이더 차트 테이블 생성
    glp_radar, glp_detail_radar, alt_radar, alt_detail_radar = create_radar_tables(leaderboard_df)
    
    # 벤치마크별 상세 테이블 생성
    benchmark_detail_df = create_benchmark_detail_table(model_name, benchmark_scores)
    
    # W&B Table로 변환 및 로깅
    try:
        wandb_run.log({"leaderboard_table": wandb.Table(dataframe=leaderboard_df)})
        print("✅ Logged: leaderboard_table")
        
        wandb_run.log({"benchmark_detail_table": wandb.Table(dataframe=benchmark_detail_df)})
        print("✅ Logged: benchmark_detail_table")
        
        if not glp_radar.empty:
            wandb_run.log({"glp_radar_table": wandb.Table(dataframe=glp_radar)})
            print("✅ Logged: glp_radar_table")
        
        if not glp_detail_radar.empty:
            wandb_run.log({"glp_detail_radar_table": wandb.Table(dataframe=glp_detail_radar)})
            print("✅ Logged: glp_detail_radar_table")
        
        if not alt_radar.empty:
            wandb_run.log({"alt_radar_table": wandb.Table(dataframe=alt_radar)})
            print("✅ Logged: alt_radar_table")
        
        if not alt_detail_radar.empty:
            wandb_run.log({"alt_detail_radar_table": wandb.Table(dataframe=alt_detail_radar)})
            print("✅ Logged: alt_detail_radar_table")
        
        # Summary에 주요 점수 추가
        if "FINAL_SCORE" in leaderboard_df.columns:
            final_score = leaderboard_df["FINAL_SCORE"].iloc[0]
            if pd.notna(final_score):
                wandb_run.summary["final_score"] = final_score
        
        if "범용언어성능(GLP)_AVG" in leaderboard_df.columns:
            glp_avg = leaderboard_df["범용언어성능(GLP)_AVG"].iloc[0]
            if pd.notna(glp_avg):
                wandb_run.summary["glp_avg"] = glp_avg
        
        if "가치정렬성능(ALT)_AVG" in leaderboard_df.columns:
            alt_avg = leaderboard_df["가치정렬성능(ALT)_AVG"].iloc[0]
            if pd.notna(alt_avg):
                wandb_run.summary["alt_avg"] = alt_avg
        
        print("✅ W&B Leaderboard tables logged successfully!")
        
    except Exception as e:
        print(f"❌ Failed to log W&B tables: {e}")
        raise


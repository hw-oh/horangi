"""
설정 로더

base_config.yaml + models/<model_name>.yaml 두 설정을 통합하여 로드합니다.

사용법:
    from core.config_loader import ConfigLoader
    
    config = ConfigLoader()
    
    # 전체 설정 로드
    full_config = config.load("gpt-4o")
    
    # 개별 접근
    wandb_config = config.wandb
    model_config = config.get_model("gpt-4o")
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """
    두 딕셔너리를 깊은 병합 (deep merge)
    
    override가 우선, 중첩된 dict는 재귀적으로 병합
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


class ConfigLoader:
    """설정 로더 클래스"""
    
    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Args:
            config_dir: configs 디렉토리 경로. None이면 자동 탐색
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # 프로젝트 루트의 configs 디렉토리 찾기
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "configs").exists():
                    self.config_dir = current / "configs"
                    break
                current = current.parent
            else:
                raise FileNotFoundError("configs 디렉토리를 찾을 수 없습니다")
        
        self._base_config: Optional[dict] = None
        self._model_configs: dict[str, dict] = {}
    
    @property
    def base_config_path(self) -> Path:
        return self.config_dir / "base_config.yaml"
    
    @property
    def models_dir(self) -> Path:
        return self.config_dir / "models"
    
    def _load_yaml(self, path: Path) -> dict:
        """YAML 파일 로드"""
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    @property
    def base(self) -> dict:
        """base_config.yaml 로드 (캐싱)"""
        if self._base_config is None:
            self._base_config = self._load_yaml(self.base_config_path)
        return self._base_config
    
    @property
    def wandb(self) -> dict:
        """WandB 설정"""
        return self.base.get("wandb", {})
    
    @property
    def defaults(self) -> dict:
        """기본 설정값"""
        return self.base.get("defaults", {})
    
    @property
    def benchmarks(self) -> dict:
        """벤치마크 공통 설정"""
        return self.base.get("benchmarks", {})
    
    @property
    def testmode(self) -> bool:
        """테스트 모드 여부"""
        return self.base.get("testmode", False)
    
    def get_model(self, model_name: str) -> dict:
        """
        모델 설정 로드
        
        Args:
            model_name: 모델 이름 (예: "gpt-4o", "claude-3-5-sonnet")
                       파일 확장자 없이 지정
        
        Returns:
            모델 설정 딕셔너리
        """
        if model_name in self._model_configs:
            return self._model_configs[model_name]
        
        # 파일명 정규화: openai/gpt-4o -> gpt-4o
        # 슬래시가 있으면 마지막 부분만 사용
        if "/" in model_name:
            file_name = model_name.split("/")[-1]
        else:
            file_name = model_name
        
        model_path = self.models_dir / f"{file_name}.yaml"
        model_config = self._load_yaml(model_path)
        
        # 캐싱
        self._model_configs[model_name] = model_config
        
        return model_config
    
    def list_models(self) -> list[str]:
        """사용 가능한 모델 설정 목록"""
        if not self.models_dir.exists():
            return []
        
        return [
            p.stem
            for p in self.models_dir.glob("*.yaml")
            if p.is_file()
        ]
    
    def load(self, model_name: Optional[str] = None) -> dict:
        """
        전체 설정 로드 (base + model 병합)
        
        Args:
            model_name: 모델 이름. None이면 base_config만 반환
        
        Returns:
            병합된 설정 딕셔너리
            {
                "wandb": {...},
                "defaults": {...},
                "benchmarks": {...},
                "testmode": bool,
                "model": {...}  # model_name이 지정된 경우
            }
        """
        config = self.base.copy()
        
        if model_name:
            model_config = self.get_model(model_name)
            config["model"] = model_config
            
            # 모델 설정의 defaults를 base defaults와 병합
            if "defaults" in model_config:
                config["defaults"] = _deep_merge(
                    config.get("defaults", {}),
                    model_config["defaults"]
                )
        
        return config
    
    def get_model_api_base(self, model_name: str) -> Optional[str]:
        """모델의 API base URL 반환"""
        model_config = self.get_model(model_name)
        return model_config.get("api_base") or model_config.get("base_url")
    
    def get_model_api_key_env(self, model_name: str) -> Optional[str]:
        """모델의 API 키 환경변수 이름 반환"""
        model_config = self.get_model(model_name)
        return model_config.get("api_key_env")
    
    def get_model_api_key(self, model_name: str) -> Optional[str]:
        """모델의 API 키 반환 (환경변수에서 읽기)"""
        env_name = self.get_model_api_key_env(model_name)
        if env_name:
            return os.environ.get(env_name)
        return None
    
    def get_inspect_model_args(self, model_name: str) -> dict:
        """
        Inspect AI 모델 생성에 필요한 인자 반환
        
        Returns:
            {
                "model": "openai/gpt-4o",
                "model_base_url": "https://...",
                ...
            }
        """
        model_config = self.get_model(model_name)
        
        args = {}
        
        # 모델 ID (provider/model 형식)
        if "model_id" in model_config:
            args["model"] = model_config["model_id"]
        else:
            args["model"] = model_name
        
        # Base URL
        base_url = self.get_model_api_base(model_name)
        if base_url:
            args["model_base_url"] = base_url
        
        # 추가 설정
        for key in ["temperature", "max_tokens", "top_p"]:
            if key in model_config:
                args[key] = model_config[key]
        
        return args


# 전역 인스턴스
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """전역 설정 로더 인스턴스 반환"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(model_name: Optional[str] = None) -> dict:
    """설정 로드 (편의 함수)"""
    return get_config().load(model_name)


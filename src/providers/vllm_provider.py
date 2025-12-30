"""
vLLM Provider for inspect_ai

This provider enables native vLLM support by:
1. Managing vLLM server lifecycle (start, health check, shutdown)
2. Providing OpenAI-compatible API interface for inspect_ai
3. Handling GPU memory cleanup after evaluation

Usage in config:
    model:
      name: Qwen/Qwen3-4B-Instruct-2507
      client: vllm
      
      vllm:
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.9
        max_model_len: 32768
        dtype: auto
        port: 8000
        trust_remote_code: true
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


@dataclass
class VLLMConfig:
    """vLLM server configuration"""
    # Model settings
    model_name_or_path: str
    dtype: str = "auto"
    max_model_len: int = 4096
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # GPU settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # Optional settings
    quantization: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    
    # Reasoning parser (for reasoning models)
    reasoning_parser: Optional[str] = None
    
    # Additional vllm serve args
    extra_args: dict = field(default_factory=dict)
    
    @property
    def base_url(self) -> str:
        """Get OpenAI-compatible API base URL"""
        return f"http://localhost:{self.port}/v1"
    
    @classmethod
    def from_dict(cls, config: dict) -> "VLLMConfig":
        """Create VLLMConfig from config dictionary"""
        model_name = config.get("model", {}).get("name", "")
        vllm_config = config.get("model", {}).get("vllm", {})
        
        return cls(
            model_name_or_path=model_name,
            dtype=vllm_config.get("dtype", "auto"),
            max_model_len=vllm_config.get("max_model_len", 4096),
            host=vllm_config.get("host", "0.0.0.0"),
            port=vllm_config.get("port", 8000),
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
            quantization=vllm_config.get("quantization"),
            revision=vllm_config.get("revision"),
            trust_remote_code=vllm_config.get("trust_remote_code", False),
            download_dir=vllm_config.get("download_dir"),
            reasoning_parser=vllm_config.get("reasoning_parser"),
            extra_args=vllm_config.get("extra_args", {}),
        )


class VLLMServerManager:
    """
    vLLM Server Manager
    
    Manages the lifecycle of a vLLM OpenAI-compatible API server.
    
    Example:
        manager = VLLMServerManager(config)
        manager.start()
        # ... run evaluations using http://localhost:8000/v1 ...
        manager.shutdown()
    
    Or use as context manager:
        with VLLMServerManager(config) as manager:
            # Server is running, use manager.base_url
            pass
        # Server is automatically shut down
    """
    
    def __init__(
        self,
        config: VLLMConfig,
        health_check_timeout: int = 300,
        health_check_interval: int = 5,
        startup_wait: int = 10,
    ):
        """
        Initialize VLLMServerManager.
        
        Args:
            config: vLLM configuration
            health_check_timeout: Maximum time to wait for server health (seconds)
            health_check_interval: Interval between health checks (seconds)
            startup_wait: Initial wait time before first health check (seconds)
        """
        self.config = config
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.startup_wait = startup_wait
        
        self._process: Optional[subprocess.Popen] = None
        self._pid_file = Path("vllm_server.pid")
        self._cleanup_registered = False
    
    @property
    def base_url(self) -> str:
        """Get the server base URL"""
        return self.config.base_url
    
    @property
    def is_running(self) -> bool:
        """Check if server process is running"""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def _build_command(self) -> list[str]:
        """Build vLLM server command"""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name_or_path,
            "--dtype", self.config.dtype,
            "--max-model-len", str(self.config.max_model_len),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--seed", "42",
            "--uvicorn-log-level", "warning",
            "--disable-log-stats",
            "--disable-log-requests",
        ]
        
        # Optional parameters
        if self.config.download_dir:
            cmd.extend(["--download-dir", self.config.download_dir])
        
        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])
        
        if self.config.revision:
            cmd.extend(["--revision", self.config.revision])
        
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if self.config.reasoning_parser:
            cmd.extend(["--reasoning-parser", self.config.reasoning_parser])
        
        # Extra args
        for key, value in self.config.extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _health_check(self) -> bool:
        """Check if the server is healthy and ready"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for health check. Install it with: pip install requests")
        
        url = f"http://localhost:{self.config.port}/health"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def _wait_for_ready(self) -> bool:
        """Wait for server to be ready"""
        print(f"⏳ Waiting for vLLM server to be ready (timeout: {self.health_check_timeout}s)...")
        
        # Initial wait
        time.sleep(self.startup_wait)
        
        start_time = time.time()
        while time.time() - start_time < self.health_check_timeout:
            if self._health_check():
                print("✅ vLLM server is ready!")
                return True
            
            # Check if process died
            if self._process and self._process.poll() is not None:
                print("❌ vLLM server process died unexpectedly")
                return False
            
            print(f"   Server not ready yet, retrying in {self.health_check_interval}s...")
            time.sleep(self.health_check_interval)
        
        print(f"❌ vLLM server failed to start within {self.health_check_timeout} seconds")
        return False
    
    def _register_cleanup(self):
        """Register cleanup handlers for graceful shutdown"""
        if self._cleanup_registered:
            return
        
        atexit.register(self.shutdown)
        
        # Handle SIGTERM
        original_handler = signal.getsignal(signal.SIGTERM)
        
        def sigterm_handler(sig, frame):
            print("\n🛑 SIGTERM received. Shutting down vLLM server...")
            self.shutdown()
            if callable(original_handler) and original_handler not in (signal.SIG_DFL, signal.SIG_IGN):
                original_handler(sig, frame)
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, sigterm_handler)
        self._cleanup_registered = True
    
    def start(self) -> bool:
        """
        Start the vLLM server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if self.is_running:
            print("⚠️ vLLM server is already running")
            return True
        
        print(f"\n{'='*60}")
        print("🚀 Starting vLLM server...")
        print(f"   Model: {self.config.model_name_or_path}")
        print(f"   Port: {self.config.port}")
        print(f"   Tensor Parallel: {self.config.tensor_parallel_size}")
        print(f"   GPU Memory Utilization: {self.config.gpu_memory_utilization}")
        print(f"{'='*60}\n")
        
        # Build command
        cmd = self._build_command()
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            # Start server process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            # Save PID
            self._pid_file.write_text(str(self._process.pid))
            
            # Register cleanup handlers
            self._register_cleanup()
            
            # Wait for server to be ready
            if self._wait_for_ready():
                print(f"\n✅ vLLM server started successfully!")
                print(f"   PID: {self._process.pid}")
                print(f"   Base URL: {self.base_url}")
                return True
            else:
                self.shutdown()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start vLLM server: {e}")
            self.shutdown()
            return False
    
    def shutdown(self):
        """Shutdown the vLLM server and clean up resources"""
        if self._process is None:
            # Try to find process from PID file
            if self._pid_file.exists():
                try:
                    pid = int(self._pid_file.read_text().strip())
                    self._terminate_process_by_pid(pid)
                except (ValueError, FileNotFoundError):
                    pass
            return
        
        print("\n🛑 Shutting down vLLM server...")
        
        try:
            # Terminate gracefully
            self._process.terminate()
            
            try:
                self._process.wait(timeout=30)
                print(f"   Process terminated gracefully (PID: {self._process.pid})")
            except subprocess.TimeoutExpired:
                print("   Termination timed out, killing process...")
                self._process.kill()
                self._process.wait()
            
            self._process = None
            
            # Remove PID file
            if self._pid_file.exists():
                self._pid_file.unlink()
                
        except Exception as e:
            print(f"   Error during shutdown: {e}")
        finally:
            # GPU memory cleanup
            self._cleanup_gpu_memory()
    
    def _terminate_process_by_pid(self, pid: int):
        """Terminate process by PID using psutil"""
        if not PSUTIL_AVAILABLE:
            print(f"   psutil not available, cannot terminate PID {pid}")
            return
        
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired:
                process.kill()
            print(f"   Terminated process with PID {pid}")
        except psutil.NoSuchProcess:
            print(f"   Process {pid} not found")
        except Exception as e:
            print(f"   Error terminating PID {pid}: {e}")
        finally:
            if self._pid_file.exists():
                self._pid_file.unlink()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after server shutdown"""
        if not TORCH_AVAILABLE:
            return
        
        print("   Cleaning up GPU memory...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                
                # Clear for all CUDA devices
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                
                print("   GPU memory cleanup completed")
        except Exception as e:
            print(f"   GPU cleanup warning: {e}")
        
        # Wait a bit for resources to be freed
        time.sleep(2)
    
    def __enter__(self) -> "VLLMServerManager":
        """Context manager entry"""
        if not self.start():
            raise RuntimeError("Failed to start vLLM server")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


# Global server instance for lifecycle management
_vllm_server: Optional[VLLMServerManager] = None


def get_vllm_server() -> Optional[VLLMServerManager]:
    """Get the global vLLM server instance"""
    return _vllm_server


def start_vllm_server(config: dict | VLLMConfig) -> VLLMServerManager:
    """
    Start a vLLM server from configuration.
    
    Args:
        config: Either a VLLMConfig object or a config dict from YAML
    
    Returns:
        VLLMServerManager instance
    
    Example:
        # From config dict
        manager = start_vllm_server(config_loader.get_model("Qwen3-4B"))
        
        # From VLLMConfig
        vllm_config = VLLMConfig(model_name_or_path="Qwen/Qwen3-4B")
        manager = start_vllm_server(vllm_config)
    """
    global _vllm_server
    
    # Shutdown existing server if any
    if _vllm_server is not None:
        _vllm_server.shutdown()
    
    # Create config
    if isinstance(config, VLLMConfig):
        vllm_config = config
    else:
        vllm_config = VLLMConfig.from_dict(config)
    
    # Create and start server
    _vllm_server = VLLMServerManager(vllm_config)
    if not _vllm_server.start():
        raise RuntimeError("Failed to start vLLM server")
    
    return _vllm_server


def shutdown_vllm_server():
    """Shutdown the global vLLM server instance"""
    global _vllm_server
    
    if _vllm_server is not None:
        _vllm_server.shutdown()
        _vllm_server = None


def is_vllm_client(config_name: str) -> bool:
    """
    Check if a model config uses vLLM client.
    
    This is a helper to determine if we need to start a vLLM server.
    """
    from core.config_loader import get_config
    
    config = get_config()
    client = config.get_model_client(config_name)
    return client == "vllm"


def get_vllm_base_url(config_name: str) -> str:
    """
    Get the vLLM server base URL for a config.
    
    If a vLLM server is running, returns its URL.
    Otherwise, returns the configured base_url from YAML.
    """
    global _vllm_server
    
    if _vllm_server is not None and _vllm_server.is_running:
        return _vllm_server.base_url
    
    from core.config_loader import get_config
    config = get_config()
    return config.get_model_base_url(config_name) or "http://localhost:8000/v1"


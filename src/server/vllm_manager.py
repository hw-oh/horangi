"""
vLLM Server Manager

Automatically start and stop vLLM OpenAI-compatible server for evaluation.
Based on wandb/llm-leaderboard vllm_server.py with improvements.

Usage:
    from server.vllm_manager import VLLMServerManager
    
    config = {
        "model_path": "LGAI-EXAONE/EXAONE-4.0.1-32B-Instruct",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "port": 8000,
    }
    
    # As context manager (recommended)
    with VLLMServerManager(config) as server:
        # Server is ready, run evaluation
        pass
    # Server is automatically stopped
    
    # Or manually
    server = VLLMServerManager(config)
    server.start()
    # ... run evaluation ...
    server.stop()
"""

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

# Optional imports for better cleanup
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for logger_name in ["httpx", "httpcore", "openai", "litellm"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def force_cuda_memory_cleanup() -> None:
    """Force cleanup of CUDA memory."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning(f"Error during CUDA memory cleanup: {e}")


def wait_for_gpu_memory_release(timeout: int = 30) -> None:
    """Wait for GPU memory to be released."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    
    start_time = time.time()
    while torch.cuda.memory_allocated() > 0:
        if time.time() - start_time > timeout:
            logger.warning(f"GPU memory not fully released after {timeout} seconds")
            break
        force_cuda_memory_cleanup()
        time.sleep(1)


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True


def find_and_kill_process_on_port(port: int) -> bool:
    """Find and kill process using a specific port."""
    if not HAS_PSUTIL:
        logger.warning("psutil not available, cannot find process on port")
        return False
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.info(f"Found process using port {port}: PID {proc.pid}")
                        kill_process_tree(proc.pid)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error while finding process on port: {e}")
    return False


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children."""
    if not HAS_PSUTIL:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return
    
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Wait for children to terminate
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        # Kill any remaining children
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Terminate parent
        try:
            parent.terminate()
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
            
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Error killing process tree: {e}")


def wait_for_port_release(port: int, timeout: int = 30) -> None:
    """Wait for a port to be released."""
    start_time = time.time()
    while is_port_in_use(port):
        if time.time() - start_time > timeout:
            # Try to forcefully kill process on port
            if find_and_kill_process_on_port(port):
                logger.info(f"Forcefully killed process using port {port}")
                time.sleep(2)
                if not is_port_in_use(port):
                    return
            raise TimeoutError(f"Port {port} was not released after {timeout} seconds")
        time.sleep(1)


def _build_vllm_command(
    model_path: str,
    port: int,
    host: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    dtype: str,
    quantization: Optional[str],
    enforce_eager: bool,
    chat_template: Optional[str],
    served_model_name: Optional[str],
    extra_args: dict[str, Any],
) -> list[str]:
    """Build vllm serve command arguments."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        # Suppress vLLM logs to keep evaluation output clean
        "--uvicorn-log-level", "warning",
        "--disable-log-stats",
        "--disable-log-requests",
    ]
    
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    
    # Add extra arguments
    for key, value in extra_args.items():
        arg_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(arg_key)
        elif value is not None:
            cmd.extend([arg_key, str(value)])
    
    return cmd


class VLLMServerManager:
    """
    Manages vLLM OpenAI-compatible server lifecycle.
    
    Starts vLLM server using subprocess and provides health check
    to wait until server is ready. Includes proper cleanup of GPU memory
    and port resources.
    """
    
    def __init__(
        self,
        config: dict[str, Any],
        startup_timeout: int = 600,
        health_check_interval: float = 2.0,
        log_dir: str = "./logs",
    ):
        """
        Args:
            config: vLLM configuration dictionary
                - model_path: HuggingFace model ID or local path (required)
                - tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
                - gpu_memory_utilization: GPU memory utilization (default: 0.9)
                - port: Server port (default: 8000)
                - host: Server host (default: "0.0.0.0")
                - max_model_len: Maximum model context length (optional)
                - trust_remote_code: Trust remote code from HuggingFace (default: True)
                - dtype: Data type (default: "auto")
                - quantization: Quantization method (optional)
                - enforce_eager: Disable CUDA graph (default: False)
                - chat_template: Path to chat template file (optional)
                - served_model_name: Model name for API (optional)
                - enable_auto_tool_choice: Enable auto tool choice (optional)
                - tool_call_parser: Tool call parser (e.g., "hermes") (optional)
                - reasoning_parser: Reasoning parser (e.g., "deepseek_r1") (optional)
            startup_timeout: Maximum time to wait for server startup (seconds)
            health_check_interval: Interval between health checks (seconds)
            log_dir: Directory to store vLLM server logs after startup
        """
        self.config = config
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.log_dir = Path(log_dir)
        
        # Extract config values with defaults
        self.model_path = config.get("model_path")
        if not self.model_path:
            raise ValueError("vllm.model_path is required")
        
        self.port = config.get("port", 8000)
        self.host = config.get("host", "0.0.0.0")
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.max_model_len = config.get("max_model_len")
        self.trust_remote_code = config.get("trust_remote_code", True)
        self.dtype = config.get("dtype", "auto")
        self.quantization = config.get("quantization")
        self.enforce_eager = config.get("enforce_eager", False)
        self.chat_template = config.get("chat_template")
        self.served_model_name = config.get("served_model_name")
        
        # Extra arguments (anything not in the above list)
        known_keys = {
            "model_path", "port", "host", "tensor_parallel_size", "gpu_memory_utilization",
            "max_model_len", "trust_remote_code", "dtype", "quantization",
            "enforce_eager", "chat_template", "served_model_name",
        }
        self.extra_args = {k: v for k, v in config.items() if k not in known_keys}
        
        self._process: Optional[subprocess.Popen] = None
        self._started = False
        self._suppress_logs = False
        self._log_threads: list[threading.Thread] = []
        self._log_file: Optional[Path] = None
        self._log_handle = None
        self._cleanup_registered = False
    
    @property
    def base_url(self) -> str:
        """Return the base URL for the vLLM server."""
        return f"http://localhost:{self.port}/v1"
    
    @property
    def health_url(self) -> str:
        """Return the health check URL."""
        return f"http://localhost:{self.port}/health"
    
    def _check_health(self) -> bool:
        """Check if the server is healthy and ready."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.health_url)
                return response.status_code == 200
        except Exception:
            return False
    
    def _stream_output(self, pipe, name: str) -> None:
        """
        Stream output from pipe to terminal during startup,
        then to log file after server is ready.
        """
        try:
            for line in iter(pipe.readline, b''):
                try:
                    decoded = line.decode('utf-8', errors='replace')
                    
                    if not self._suppress_logs:
                        # During startup: print to terminal
                        print(decoded, end='', flush=True)
                    
                    # Always write to log file if available
                    if self._log_handle:
                        self._log_handle.write(decoded)
                        self._log_handle.flush()
                        
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass
    
    def start(self) -> None:
        """
        Start the vLLM server.
        
        Raises:
            RuntimeError: If server is already running or fails to start
            TimeoutError: If server doesn't become ready within timeout
        """
        if self._started:
            raise RuntimeError("vLLM server is already running")
        
        # Pre-startup cleanup
        if is_port_in_use(self.port):
            # Check if it's already a vLLM server we can use
            if self._check_health():
                logger.info(f"vLLM server already running on port {self.port}, using existing server")
                print(f"⚠️ vLLM server already running on port {self.port}, using existing server")
                self._started = True
                self._process = None  # Don't manage external server
                return
            
            # Try to clean up the port
            logger.info(f"Port {self.port} is in use. Attempting to clean up...")
            print(f"⚠️ Port {self.port} is in use. Attempting to clean up...")
            find_and_kill_process_on_port(self.port)
            try:
                wait_for_port_release(self.port, timeout=15)
            except TimeoutError:
                raise RuntimeError(f"Port {self.port} is already in use and could not be released")
        
        # Clean up GPU memory before starting
        force_cuda_memory_cleanup()
        
        # Create log directory and file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "vllm_server.log"
        self._log_handle = open(self._log_file, "w")
        
        print(f"🚀 Starting vLLM server...")
        print(f"   Model: {self.model_path}")
        print(f"   Host: {self.host}:{self.port}")
        print(f"   Tensor Parallel: {self.tensor_parallel_size}")
        print(f"   GPU Memory: {self.gpu_memory_utilization}")
        if self.served_model_name:
            print(f"   Served Model Name: {self.served_model_name}")
        if self.extra_args:
            print(f"   Extra Args: {self.extra_args}")
        print(f"")
        print(f"{'='*60}")
        print(f"vLLM Server Startup Logs:")
        print(f"{'='*60}")
        
        # Build command
        cmd = _build_vllm_command(
            model_path=self.model_path,
            port=self.port,
            host=self.host,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            chat_template=self.chat_template,
            served_model_name=self.served_model_name,
            extra_args=self.extra_args,
        )
        
        # Log command for debugging
        logger.debug(f"vLLM command: {' '.join(cmd)}")
        
        # Start server with PIPE to control output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        
        # Save PID for potential recovery
        pid_file = self.log_dir / "vllm_server.pid"
        with open(pid_file, "w") as f:
            f.write(str(self._process.pid))
        
        # Start log streaming threads (shows output during startup)
        self._suppress_logs = False
        
        stdout_thread = threading.Thread(
            target=self._stream_output,
            args=(self._process.stdout, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._stream_output,
            args=(self._process.stderr, "stderr"),
            daemon=True,
        )
        
        stdout_thread.start()
        stderr_thread.start()
        self._log_threads = [stdout_thread, stderr_thread]
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.stop)
            self._cleanup_registered = True
        
        # Wait for server to be ready
        self._wait_ready()
        self._started = True
        
        # Suppress logs after server is ready
        self._suppress_logs = True
        
        print(f"{'='*60}")
        print(f"✅ vLLM server ready at {self.base_url}")
        print(f"   (Runtime logs suppressed, saved to: {self._log_file})")
        print(f"{'='*60}\n")
    
    def _wait_ready(self) -> None:
        """
        Wait for the server to become ready.
        
        Raises:
            TimeoutError: If server doesn't become ready within timeout
            RuntimeError: If server process dies
        """
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            # Check if process is still alive
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"vLLM server process died with exit code {self._process.returncode}")
            
            # Check health
            if self._check_health():
                return
            
            time.sleep(self.health_check_interval)
        
        # Timeout reached
        self.stop()
        raise TimeoutError(
            f"vLLM server failed to start within {self.startup_timeout} seconds"
        )
    
    def stop(self) -> None:
        """Stop the vLLM server and clean up resources."""
        if self._process is None:
            # External server or not started
            self._started = False
            return
        
        if self._process.poll() is None:  # Process is still running
            print(f"\n🛑 Stopping vLLM server...")
            
            if HAS_PSUTIL:
                # Use psutil for proper process tree cleanup
                kill_process_tree(self._process.pid)
            else:
                # Fallback to process group kill
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                    try:
                        self._process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        print(f"   ⚠️ Force killing server...")
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                        self._process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    print(f"   ⚠️ Error stopping server: {e}")
            
            print(f"✅ vLLM server stopped")
        
        # Close log file handle
        if self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        
        # Clean up PID file
        pid_file = self.log_dir / "vllm_server.pid"
        if pid_file.exists():
            try:
                pid_file.unlink()
            except Exception:
                pass
        
        self._process = None
        self._started = False
        self._suppress_logs = True
        
        # Wait for port to be released
        try:
            wait_for_port_release(self.port, timeout=15)
        except TimeoutError:
            logger.warning(f"Port {self.port} could not be released")
        
        # Clean up GPU memory
        logger.info("Cleaning up GPU memory...")
        force_cuda_memory_cleanup()
        wait_for_gpu_memory_release(timeout=30)
        logger.info("GPU memory cleanup completed.")
        
        # Brief wait for cleanup to complete
        time.sleep(2)
        
        # Unregister cleanup
        if self._cleanup_registered:
            try:
                atexit.unregister(self.stop)
                self._cleanup_registered = False
            except Exception:
                pass
    
    def __enter__(self) -> "VLLMServerManager":
        """Context manager entry - start the server."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the server."""
        self.stop()
    
    def is_running(self) -> bool:
        """Check if the server is running and healthy."""
        if not self._started:
            return False
        return self._check_health()

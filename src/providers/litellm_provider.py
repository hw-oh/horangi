"""
LiteLLM Provider for inspect_ai

This provider uses LiteLLM client library to call various LLM providers.
Benefits:
- Unified interface for all providers
- Better Weave token tracking (uses OpenAI-compatible format)
- Supports LiteLLM's extended features

Usage in config:
    model_id: litellm/anthropic/claude-opus-4-5-20251101
    # or
    model_id: litellm/openai/gpt-4
"""

import os
from typing import Any

from inspect_ai.model import ModelAPI, modelapi, GenerateConfig, ModelOutput, ChatCompletionChoice
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._model_output import ModelUsage, StopReason
from inspect_ai.model._model_call import ModelCall
from inspect_ai.tool import ToolChoice, ToolInfo

try:
    import litellm
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def _stop_reason_from_litellm(finish_reason: str | None) -> StopReason:
    """Convert LiteLLM finish_reason to inspect_ai StopReason."""
    if finish_reason is None:
        return "unknown"
    mapping = {
        "stop": "stop",
        "length": "max_tokens",
        "content_filter": "content_filter",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
    }
    return mapping.get(finish_reason, "unknown")


@modelapi(name="litellm")
def litellm_api() -> type["LiteLLMAPI"]:
    """Register LiteLLM as a model API provider."""
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "LiteLLM is not installed. Install it with: pip install litellm"
        )
    return LiteLLMAPI


class LiteLLMAPI(ModelAPI):
    """
    LiteLLM Model API for inspect_ai.
    
    Supports all LiteLLM providers including:
    - anthropic/claude-*
    - openai/gpt-*
    - together_ai/*
    - etc.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        
        # Store model args for LiteLLM-specific features
        self._model_args = model_args
        
        # Extract the actual model name (remove 'litellm/' prefix if present)
        if model_name.startswith("litellm/"):
            self._litellm_model = model_name[8:]  # Remove 'litellm/' prefix
        else:
            self._litellm_model = model_name
        
        # Set up LiteLLM configuration to avoid event loop issues
        litellm.set_verbose = False
        litellm.suppress_debug_info = True
        litellm.drop_params = True  # Drop unsupported params instead of error
        
        # Disable async logging to prevent "bound to different event loop" error
        litellm.disable_logging = True
        litellm.turn_off_message_logging = True
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        
        # Disable the logging worker completely
        if hasattr(litellm, '_logging_worker') and litellm._logging_worker is not None:
            try:
                litellm._logging_worker = None
            except Exception:
                pass
        
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput, ModelCall]:
        """Generate a response using LiteLLM."""
        
        # Convert messages to OpenAI format
        messages = self._convert_messages(input)
        
        # Build request parameters
        params: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": messages,
        }
        
        # Pass all non-None config attributes directly to LiteLLM
        # Use model_dump() for Pydantic models to get actual values (not FieldInfo)
        try:
            config_dict = config.model_dump(exclude_none=True)
        except AttributeError:
            # Fallback for non-Pydantic objects
            config_dict = {k: v for k, v in vars(config).items() if v is not None and not k.startswith('_')}
        
        for attr, value in config_dict.items():
            if callable(value):
                continue
            # Special case: reasoning_tokens -> thinking (Extended Thinking)
            if attr == "reasoning_tokens":
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": value
                }
            # Special case: stop_seqs -> stop
            elif attr == "stop_seqs":
                params["stop"] = value
            # Skip internal/unsupported params
            elif attr in ("cache", "internal_tools", "max_tool_output"):
                continue
            else:
                params[attr] = value
        
        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice != "auto":
                params["tool_choice"] = self._convert_tool_choice(tool_choice)
        
        # Pass through any additional model_args as **kwargs
        for key, value in self._model_args.items():
            if value is not None:
                params[key] = value
        
        # Handle timeout parameter for litellm
        timeout = params.pop("timeout", None)
        if timeout is not None:
            params["timeout"] = float(timeout)
        
        # Make the API call
        response = await acompletion(**params)
        
        # Convert response to ModelOutput
        return self._convert_response(response, tools)
    
    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert inspect_ai messages to OpenAI format."""
        result = []
        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.text})
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.text})
            elif msg.role == "assistant":
                content = msg.text if hasattr(msg, 'text') else str(msg.content)
                result.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id if hasattr(msg, 'tool_call_id') else "",
                    "content": msg.text if hasattr(msg, 'text') else str(msg.content),
                })
        return result
    
    def _convert_tools(self, tools: list[ToolInfo]) -> list[dict[str, Any]]:
        """Convert inspect_ai tools to OpenAI format."""
        result = []
        for tool in tools:
            # Convert parameters to dict if it's a Pydantic model
            params = tool.parameters
            if hasattr(params, 'model_dump'):
                params = params.model_dump(exclude_none=True)
            elif hasattr(params, 'dict'):
                params = params.dict(exclude_none=True)
            
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                }
            })
        return result
    
    def _convert_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any] | str:
        """Convert inspect_ai tool_choice to OpenAI format."""
        if tool_choice == "none":
            return "none"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "any":
            return "required"
        elif isinstance(tool_choice, dict) and "name" in tool_choice:
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        elif hasattr(tool_choice, 'name'):
            # Handle ToolFunction Pydantic model
            return {"type": "function", "function": {"name": tool_choice.name}}
        return "auto"
    
    def _convert_response(self, response: Any, tools: list[ToolInfo]) -> ModelOutput:
        """Convert LiteLLM response to ModelOutput."""
        choice = response.choices[0]
        message = choice.message
        
        # Build assistant message
        content = message.content or ""
        
        # Handle tool calls
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        
        assistant_message = ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls,
            source="generate",
        )
        
        # Build usage
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = ModelUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        
        # Build completion choice
        completion_choice = ChatCompletionChoice(
            message=assistant_message,
            stop_reason=_stop_reason_from_litellm(choice.finish_reason),
        )
        
        return ModelOutput(
            model=response.model,
            choices=[completion_choice],
            usage=usage,
        )
    
    def max_tokens(self) -> int | None:
        """Return default max tokens."""
        return 4096
    
    def connection_key(self) -> str:
        """Return a unique key for connection pooling."""
        return f"litellm:{self._litellm_model}"


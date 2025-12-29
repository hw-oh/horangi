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
        
        # Set up LiteLLM configuration
        litellm.set_verbose = False
        
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
        
        # Add generation config
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.stop_seqs is not None:
            params["stop"] = config.stop_seqs
        
        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice != "auto":
                params["tool_choice"] = self._convert_tool_choice(tool_choice)
        
        # Add LiteLLM-specific parameters from config
        # Reference: https://docs.litellm.ai/docs/providers/anthropic
        
        # 1. reasoning_effort - LiteLLM auto-maps to output_config.effort for Claude Opus 4.5
        if hasattr(config, 'reasoning_effort') and config.reasoning_effort is not None:
            params["reasoning_effort"] = config.reasoning_effort
        
        # 2. effort parameter (Claude Opus 4.5) - also try direct effort
        if hasattr(config, 'effort') and config.effort is not None:
            params["reasoning_effort"] = config.effort
        
        # 3. reasoning_tokens -> thinking (Extended Thinking for Sonnet 4.5, etc.)
        if hasattr(config, 'reasoning_tokens') and config.reasoning_tokens is not None:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.reasoning_tokens
            }
        
        # 4. Support model_args overrides
        if "thinking" in self._model_args:
            params["thinking"] = self._model_args["thinking"]
        if "budget_tokens" in self._model_args:
            if "thinking" not in params:
                params["thinking"] = {"type": "enabled"}
            params["thinking"]["budget_tokens"] = self._model_args["budget_tokens"]
        if "reasoning_effort" in self._model_args:
            params["reasoning_effort"] = self._model_args["reasoning_effort"]
        
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
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in tools
        ]
    
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


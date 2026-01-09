"""
MT-Bench Solver - 2-turn conversation processing

MT-Bench uses a 2-turn conversation format:
1. Turn 1: Response to the first question
2. Turn 2: Response to the second question based on the first response
"""

from inspect_ai.solver import Solver, solver, Generate, TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


@solver
def mtbench_solver() -> Solver:
    """
    MT-Bench 2-turn conversation Solver
    
    Turn 1: state.input (first question)
    Turn 2: metadata["turn2"] (second question)
    
    Results are saved in metadata as response_turn1 and response_turn2.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        
        # Turn 1: Get from state.input (field_mapping["input"] = "turn1")
        turn1 = state.input if isinstance(state.input, str) else str(state.input)
        turn2 = metadata.get("turn2", "")
        
        # Save turn1 to metadata (used by scorer)
        state.metadata["turn1"] = turn1
        
        # Turn 1: First question
        state.messages = [ChatMessageUser(content=turn1)]
        state = await generate(state)
        
        # Save Turn 1 response
        response_turn1 = state.output.completion if state.output else ""
        state.metadata["response_turn1"] = response_turn1
        
        # Continue if Turn 2 exists
        if turn2:
            # Build Turn 2 conversation history
            # Note: generate() may have already added assistant message to state.messages
            # So we rebuild the message list to ensure correct alternation
            state.messages = [
                ChatMessageUser(content=turn1),
                ChatMessageAssistant(content=response_turn1),
                ChatMessageUser(content=turn2)
            ]
            
            # Generate Turn 2
            state = await generate(state)
            response_turn2 = state.output.completion if state.output else ""
            state.metadata["response_turn2"] = response_turn2
        else:
            state.metadata["response_turn2"] = ""
        
        return state
    
    return solve

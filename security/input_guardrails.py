import logging
from llama_index.llms.ollama import Ollama
from config import ENABLE_INPUT_GUARDRAILS, GUARDRAIL_MODEL

logger = logging.getLogger(__name__)

def validate_input(query: str) -> tuple[bool, str]:
    """
    Checks the user query using Llama Guard for prompt injections or policy violations.
    
    Returns:
        tuple[bool, str]: (Is safe?, Reason/Category if unsafe)
    """
    if not ENABLE_INPUT_GUARDRAILS:
        return True, ""
        
    try:
        # We explicitly instantiate the guardrail model defined in config
        guard_llm = Ollama(model=GUARDRAIL_MODEL, request_timeout=60.0, temperature=0.0)
        
        # Llama Guard 3 expects a specific prompt format, usually provided automatically
        # by the HuggingFace integration, but Ollama expects raw text interactions.
        # We wrap the user query in the required format.
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        response = guard_llm.complete(prompt)
        result_text = str(response).strip().lower()
        
        if result_text.startswith("safe"):
            return True, ""
        elif result_text.startswith("unsafe"):
            # Llama Guard outputs "unsafe\nO1" or similar
            parts = result_text.split()
            reason = parts[1] if len(parts) > 1 else "Unknown Policy Violation"
            return False, reason
        else:
            # Fallback if the model outputs something unexpected
            logger.warning(f"Llama Guard produced unexpected output: {result_text}")
            return True, "Unexpected response from guardrail"
            
    except Exception as e:
        logger.error(f"Error during input validation: {e}")
        # Fail open or fail closed? We fail open here to not break the pipeline on timeout.
        return True, f"Error: {str(e)}"

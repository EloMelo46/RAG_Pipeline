import logging
from llama_index.llms.ollama import Ollama
from config import ENABLE_OUTPUT_GUARDRAILS, GUARDRAIL_MODEL

logger = logging.getLogger(__name__)

def validate_output(query: str, response_text: str) -> tuple[bool, str]:
    """
    Checks the LLM generated response using Llama Guard for policy violations.
    
    Returns:
        tuple[bool, str]: (Is safe?, Reason/Category if unsafe)
    """
    if not ENABLE_OUTPUT_GUARDRAILS:
        return True, ""
        
    try:
        guard_llm = Ollama(model=GUARDRAIL_MODEL, request_timeout=60.0, temperature=0.0)
        
        # We pass both the original query and the assistant's response to Llama Guard
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response_text}<|eot_id|>"""
        
        response = guard_llm.complete(prompt)
        result_text = str(response).strip().lower()
        
        if result_text.startswith("safe"):
            return True, ""
        elif result_text.startswith("unsafe"):
            parts = result_text.split()
            reason = parts[1] if len(parts) > 1 else "Unknown Policy Violation"
            return False, reason
        else:
            logger.warning(f"Llama Guard produced unexpected output: {result_text}")
            return True, "Unexpected response from guardrail"
            
    except Exception as e:
        logger.error(f"Error during output validation: {e}")
        return True, f"Error: {str(e)}"

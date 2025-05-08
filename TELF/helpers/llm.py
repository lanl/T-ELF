import json
import re
import logging
from langchain_ollama import OllamaLLM
import subprocess

log = logging.getLogger(__name__)
from typing import Iterable

def build_json_vote_prompt(candidate: str, contexts: Iterable[str]) -> str:
    """
    Build a JSON-only prompt from example contexts and a candidate string.
    """
    ctx_block = "\n----\n".join(contexts)
    return (
        "You are an expert researcher. Output ONLY valid JSON.\n"
        f"Target context examples:\n{ctx_block}\n\n"
        f"Candidate abstract:\n{candidate}\n"
        "Given the context, is the candidate about any of the concepts? "
        'Respond {"answer":"yes|no","reason":"..."}'
    )


def get_ollama_llm(model: str, base_url: str, temperature: float) -> OllamaLLM:
    """
    Create and return a configured OllamaLLM instance.
    If `model` isn’t yet pulled locally, this will shell out to:
        ollama pull <model>
    """
    try:
        available = subprocess.check_output(
            ["ollama", "list"], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Could not list Ollama models: {e}")

    if model not in available:
        try:
            print(f"Model '{model}' not found locally – pulling…")
            subprocess.run(
                ["ollama", "pull", model],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull Ollama model '{model}': {e.stderr.strip()}")

    return OllamaLLM(model=model, base_url=base_url, temperature=temperature)



def vote_once(llm: OllamaLLM, prompt: str) -> tuple[bool, str]:
    """
    Invoke the given OllamaLLM instance once, strip markdown fences,
    parse its JSON response, and return (yes_flag, reason).
    """
    raw = llm.invoke(prompt)
    txt = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        obj = json.loads(txt)
        yes = obj.get("answer", "").lower() == "yes"
        reason = str(obj.get("reason", "")).strip()
        return yes, reason
    except Exception:
        log.warning("Bad JSON from LLM: %s", raw)
        return False, ""
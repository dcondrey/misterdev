from typing import Dict, Any

from my_project_orchestrator.logging_setup import setup_logger

logger = setup_logger(__name__)


class PromptManager:
    """Manages prompt template formatting with safe variable substitution.

    Uses {{variable}} syntax instead of Python's {variable} to avoid
    conflicts with code content, JSON, and LLM output that contain braces.
    Falls back to {variable} syntax for backward compatibility with
    existing project.yaml templates.
    """

    def __init__(self, config: dict):
        self.config = config
        self.templates = config.get("prompt_templates", {})

    def format_prompt(self, template_key: str, context_dict: Dict[str, Any]) -> str:
        template = self.templates.get(template_key)
        if not template:
            raise ValueError(
                f"Prompt template '{template_key}' not found in configuration."
            )

        # Inject inherited system prompt if not already in context
        if "inherited_system_prompt" not in context_dict and template_key != "system":
            context_dict["inherited_system_prompt"] = self.templates.get("system", "")

        # Convert all context values to strings
        str_context = {
            k: str(v) if v is not None else "" for k, v in context_dict.items()
        }

        # First pass: replace {{var}} syntax (preferred, safe)
        result = template
        for key, value in str_context.items():
            result = result.replace("{{" + key + "}}", value)

        # Second pass: replace {var} syntax for backward compatibility.
        # Only replace known variables to avoid breaking code content.
        result = _safe_format(result, str_context)

        return result


def _safe_format(template: str, context: Dict[str, str]) -> str:
    """Replace {var} placeholders without breaking unrelated braces.

    Scans character-by-character for {identifier} patterns where identifier
    is a known context key. Leaves all other brace patterns untouched.
    """
    result = []
    i = 0
    length = len(template)

    while i < length:
        if template[i] != "{":
            result.append(template[i])
            i += 1
            continue

        # Found '{', try to read an identifier (word chars and dots)
        j = i + 1
        while j < length and (template[j].isalnum() or template[j] in ("_", ".")):
            j += 1

        # Check if it closes with '}'
        if j < length and template[j] == "}" and j > i + 1:
            key = template[i + 1 : j]
            if key in context:
                result.append(context[key])
                i = j + 1
                continue

        # Not a known variable; emit the '{' literally
        result.append(template[i])
        i += 1

    return "".join(result)

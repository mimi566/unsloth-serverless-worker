from transformers import AutoTokenizer
import os
from typing import Union, List, Dict


class TokenizerWrapper:
    def __init__(self, tokenizer_name_or_path, tokenizer_revision, trust_remote_code):
        print(
            f"[Tokenizer Init] name={tokenizer_name_or_path}, "
            f"revision={tokenizer_revision}, trust_remote_code={trust_remote_code}"
        )

        # Load tokenizer safely
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            revision=tokenizer_revision or "main",
            trust_remote_code=bool(trust_remote_code)
        )

        # Load template from env if provided
        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")

        # Detect if tokenizer has template
        tok_template_exists = hasattr(self.tokenizer, "chat_template") and bool(self.tokenizer.chat_template)

        # Apply custom template if available
        if self.custom_chat_template:
            print("[Tokenizer] Using CUSTOM_CHAT_TEMPLATE from environment")
            self.tokenizer.chat_template = self.custom_chat_template
            self.has_chat_template = True
        else:
            self.has_chat_template = tok_template_exists

        print(f"[Tokenizer] has_chat_template: {self.has_chat_template}")

    def apply_chat_template(self, input: Union[str, List[Dict[str, str]]]) -> str:
        """
        Converts messages into a single chat-formatted string.
        Required for vLLM when applying chat template manually.
        """

        # Convert plain text → user message list
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        if not isinstance(input, list):
            raise ValueError("Input must be a string or a list of message dicts")

        if not self.has_chat_template:
            raise ValueError(
                "This model does not have a chat template. "
                "You must provide raw text (string), not a messages[] array."
            )

        # Apply template (no tokenization, only formatting)
        try:
            formatted = self.tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            raise RuntimeError(f"Error applying chat template: {e}")

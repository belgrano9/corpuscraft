from __future__ import annotations

import json
import math

from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpuscraft.config import GeneratorConfig, LLMConfig
from corpuscraft.generators.base import BaseGenerator
from corpuscraft.models import ParsedDocument, QAExample

_SYSTEM_PROMPT = """\
You are an expert at creating question-answer pairs for training AI models.
Given a passage of text, generate diverse question-answer pairs that cover:
- Factual questions (specific facts directly stated in the text)
- Reasoning questions (require inference or synthesis)
- Clarification questions (about terminology or concepts)

Return a JSON array of objects with keys: "question", "answer", "difficulty".
Difficulty must be one of: "easy", "medium", "hard".
Return ONLY the JSON array, no other text."""

_USER_TEMPLATE = """\
Generate {n} question-answer pairs from the following passage:

{passage}"""


class QAGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig, llm: LLMConfig) -> None:
        super().__init__(config, llm)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )

    def generate(self, document: ParsedDocument) -> list[QAExample]:
        import ollama

        chunks = self._splitter.split_text(document.content)
        if not chunks:
            return []

        per_chunk = max(1, math.ceil(self.config.num_examples / len(chunks)))
        examples: list[QAExample] = []

        client = ollama.Client(host=self.llm.base_url)

        for chunk in chunks:
            if len(examples) >= self.config.num_examples:
                break
            try:
                response = client.chat(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _USER_TEMPLATE.format(
                                n=per_chunk, passage=chunk
                            ),
                        },
                    ],
                    options={"temperature": self.llm.temperature},
                )
                raw = response.message.content.strip()
                pairs = json.loads(raw)
                for pair in pairs:
                    examples.append(
                        QAExample(
                            question=pair["question"],
                            answer=pair["answer"],
                            context=chunk,
                            source=str(document.source_path),
                            difficulty=pair.get("difficulty", "medium"),
                        )
                    )
            except (json.JSONDecodeError, KeyError, Exception):
                continue

        return examples[: self.config.num_examples]

"""Question-Answering dataset generator."""

import json
import logging
import random
from typing import Any

from corpuscraft.generators.base import BaseGenerator
from corpuscraft.llm.base import BaseLLM
from corpuscraft.parsers.docling_parser import ParsedDocument

logger = logging.getLogger(__name__)


class QAGenerator(BaseGenerator):
    """Generate Question-Answer pairs from documents."""

    QA_GENERATION_PROMPT = """You are a helpful assistant that generates high-quality question-answer pairs from text passages.

Given the following text passage, generate {num_questions} diverse question-answer pairs.

PASSAGE:
{passage}

REQUIREMENTS:
- Questions should be {difficulty} difficulty
- Question types: {question_types}
- Answers should be factual and based solely on the passage
- Answer length: {min_length}-{max_length} words
- Generate diverse questions covering different aspects of the passage

OUTPUT FORMAT (JSON):
[
  {{
    "question": "the question text",
    "answer": "the answer text",
    "difficulty": "easy|medium|hard",
    "type": "factual|reasoning|comparison"
  }}
]

Generate the QA pairs now:"""

    def __init__(
        self,
        llm: BaseLLM,
        question_types: list[str] | None = None,
        difficulty_levels: list[str] | None = None,
        min_answer_length: int = 1,
        max_answer_length: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize QA generator.

        Args:
            llm: LLM backend
            question_types: Types of questions (factual, reasoning, comparison, etc.)
            difficulty_levels: Difficulty levels (easy, medium, hard)
            min_answer_length: Minimum answer length in words
            max_answer_length: Maximum answer length in words
            **kwargs: Additional parameters
        """
        super().__init__(llm, **kwargs)

        self.question_types = question_types or ["factual", "reasoning", "comparison"]
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length

    def generate(
        self,
        documents: list[ParsedDocument],
        num_examples: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate QA pairs from documents.

        Args:
            documents: List of parsed documents
            num_examples: Number of QA pairs to generate
            **kwargs: Additional parameters

        Returns:
            List of QA examples
        """
        logger.info(f"Generating {num_examples} QA pairs from {len(documents)} documents")

        # Collect all chunks from all documents
        all_chunks: list[tuple[str, ParsedDocument]] = []
        for doc in documents:
            for chunk in doc.chunks:
                all_chunks.append((chunk, doc))

        if not all_chunks:
            logger.warning("No chunks available for generation")
            return []

        # Calculate how many questions per chunk
        num_chunks = min(len(all_chunks), num_examples)
        selected_chunks = random.sample(all_chunks, num_chunks)

        qa_pairs: list[dict[str, Any]] = []
        questions_per_chunk = max(1, num_examples // num_chunks)

        for chunk, doc in selected_chunks:
            try:
                # Randomly select difficulty and type
                difficulty = random.choice(self.difficulty_levels)
                qtype = random.choice(self.question_types)

                # Generate QA pairs for this chunk
                chunk_pairs = self._generate_from_chunk(
                    chunk=chunk,
                    document=doc,
                    num_questions=questions_per_chunk,
                    difficulty=difficulty,
                    question_types=[qtype],
                )

                qa_pairs.extend(chunk_pairs)

                if len(qa_pairs) >= num_examples:
                    break

            except Exception as e:
                logger.error(f"Error generating QA from chunk: {e}")
                continue

        # Trim to exact number
        qa_pairs = qa_pairs[:num_examples]

        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        return qa_pairs

    def _generate_from_chunk(
        self,
        chunk: str,
        document: ParsedDocument,
        num_questions: int,
        difficulty: str,
        question_types: list[str],
    ) -> list[dict[str, Any]]:
        """Generate QA pairs from a single chunk.

        Args:
            chunk: Text chunk
            document: Source document
            num_questions: Number of questions to generate
            difficulty: Difficulty level
            question_types: Question types to generate

        Returns:
            List of QA pairs
        """
        # Build prompt
        prompt = self.QA_GENERATION_PROMPT.format(
            num_questions=num_questions,
            passage=chunk,
            difficulty=difficulty,
            question_types=", ".join(question_types),
            min_length=self.min_answer_length,
            max_length=self.max_answer_length,
        )

        # Generate
        response = self.llm.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block markers
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response

            qa_data = json.loads(response)

            # Add metadata to each QA pair
            for qa in qa_data:
                qa["context"] = chunk
                qa["source_file"] = document.file_path.name
                qa["source_metadata"] = document.metadata

            return qa_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            return []

    def get_example_schema(self) -> dict[str, type]:
        """Get the schema of generated examples.

        Returns:
            Dictionary mapping field names to types
        """
        return {
            "question": str,
            "answer": str,
            "context": str,
            "difficulty": str,
            "type": str,
            "source_file": str,
            "source_metadata": dict,
        }

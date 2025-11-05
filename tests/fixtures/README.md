# Test Fixtures

This directory contains sample files used for testing CorpusCraft.

## Files

- **sample_document.txt**: A sample text document about machine learning fundamentals
- **sample_config.yaml**: A sample configuration file for CorpusCraft
- **sample_qa_output.jsonl**: Example QA pairs output in JSONL format

## Usage

These fixtures are used by the test suite to verify functionality without requiring:
- Real document parsing (mocked with test content)
- Actual LLM API calls (mocked responses)
- External dependencies

Test files should reference these fixtures when appropriate to ensure consistent test behavior.

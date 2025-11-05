# CorpusCraft Test Suite

This directory contains the comprehensive test suite for CorpusCraft, covering unit tests, integration tests, and fixtures.

## Test Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures and configuration
├── fixtures/                      # Test data and sample files
│   ├── sample_document.txt       # Sample text document
│   ├── sample_config.yaml        # Sample configuration
│   └── sample_qa_output.jsonl    # Sample output format
├── test_config.py                # Tests for configuration models
├── test_llm_ollama.py            # Tests for Ollama LLM backend
├── test_parsers_docling.py       # Tests for document parser
├── test_generators_qa.py         # Tests for QA generator
├── test_output_jsonl.py          # Tests for JSONL writer
└── test_integration.py           # End-to-end integration tests
```

## Running Tests

### Install Development Dependencies

First, install the development dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- black (code formatting)
- ruff (linting)
- mypy (type checking)

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Run only configuration tests
pytest tests/test_config.py

# Run only LLM tests
pytest tests/test_llm_ollama.py

# Run integration tests
pytest tests/test_integration.py
```

### Run Tests with Coverage

```bash
pytest --cov=corpuscraft --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Specific Test Functions

```bash
# Run a specific test class
pytest tests/test_config.py::TestInputConfig

# Run a specific test method
pytest tests/test_config.py::TestInputConfig::test_default_values
```

## Test Categories

### Unit Tests

Tests for individual components in isolation:

- **test_config.py**: Configuration validation and Pydantic models
- **test_llm_ollama.py**: Ollama LLM backend with mocked API calls
- **test_parsers_docling.py**: Document parsing with mocked Docling
- **test_generators_qa.py**: QA generation logic with mocked LLM
- **test_output_jsonl.py**: JSONL writing and data splitting

### Integration Tests

End-to-end tests that verify the complete pipeline:

- **test_integration.py**: Full pipeline from parsing to output
  - Complete QA generation workflow
  - Error handling across components
  - Data flow and metadata preservation
  - Multiple generator configurations

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_text`: Sample text content
- `sample_chunks`: Pre-chunked text samples
- `sample_parsed_document`: Mock parsed document
- `sample_qa_pairs`: Example QA pair data
- `mock_ollama_client`: Mocked Ollama client
- Configuration fixtures: `input_config`, `processing_config`, `llm_config`, etc.

## Mocking Strategy

The tests use extensive mocking to avoid external dependencies:

1. **Ollama API**: Mocked using `unittest.mock.MagicMock`
2. **Docling Parser**: Mocked DocumentConverter to avoid parsing real files
3. **File System**: Uses pytest's `tmp_path` fixture for temporary directories

This ensures tests:
- Run quickly without network calls
- Don't require Ollama to be running
- Don't need real documents
- Are reproducible and isolated

## Coverage Goals

We aim for:
- **>90% overall coverage**
- **100% coverage** for critical paths:
  - Configuration validation
  - Data output formatting
  - Error handling

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

### Example Test Structure

```python
from unittest.mock import MagicMock
import pytest

class TestMyComponent:
    """Tests for MyComponent class."""

    def test_initialization(self) -> None:
        """Test component initialization."""
        component = MyComponent(param="value")
        assert component.param == "value"

    def test_error_handling(self) -> None:
        """Test that errors are handled correctly."""
        component = MyComponent()
        with pytest.raises(ValueError):
            component.invalid_operation()

    def test_with_mock(self) -> None:
        """Test with mocked dependency."""
        mock_dependency = MagicMock()
        mock_dependency.method.return_value = "result"

        component = MyComponent(dependency=mock_dependency)
        result = component.do_something()

        assert result == "processed_result"
        mock_dependency.method.assert_called_once()
```

### Using Fixtures

```python
def test_with_fixtures(
    sample_text: str,
    temp_dir: Path,
    mock_ollama_client: MagicMock,
) -> None:
    """Test using shared fixtures."""
    # Fixtures are automatically provided by pytest
    assert len(sample_text) > 0
    assert temp_dir.exists()
```

## Continuous Integration

Tests are designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest --cov=corpuscraft --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Debugging Tests

### Run Tests with Print Statements

```bash
pytest -s
```

### Run Tests with PDB on Failure

```bash
pytest --pdb
```

### Show Local Variables on Failure

```bash
pytest -l
```

### Run Only Failed Tests

```bash
pytest --lf
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (APIs, file system when possible)
3. **Fixtures**: Use fixtures for common test data
4. **Assertions**: Use clear, descriptive assertions
5. **Documentation**: Add docstrings explaining what each test verifies
6. **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple inputs
7. **Error Cases**: Test both success and error scenarios

## Common Issues

### Tests Fail Due to Missing Dependencies

```bash
# Install all dependencies
pip install -e ".[dev]"
```

### Tests Fail Due to Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .
```

### Tests Are Slow

Most tests use mocking to run quickly. If tests are slow:
- Check for accidental real API calls
- Ensure Docling is properly mocked
- Use `pytest -v` to identify slow tests

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add tests for additional generator types
- [ ] Add tests for cloud LLM backends
- [ ] Add property-based tests with Hypothesis
- [ ] Add mutation testing with mutmut

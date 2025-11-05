"""CLI interface for CorpusCraft."""

import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from corpuscraft import __version__
from corpuscraft.config import (
    CorpusCraftConfig,
    GeneratorConfig,
    GeneratorType,
    InputConfig,
    LLMBackend,
    LLMConfig,
    OutputConfig,
    ProcessingConfig,
    QAGeneratorConfig,
)
from corpuscraft.generators.qa_generator import QAGenerator
from corpuscraft.llm.ollama import OllamaLLM
from corpuscraft.output.jsonl_writer import JSONLWriter
from corpuscraft.parsers.docling_parser import DoclingParser

# Initialize CLI app
app = typer.Typer(
    name="corpuscraft",
    help="Transform your documents into training datasets",
    add_completion=False,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)

logger = logging.getLogger("corpuscraft")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"CorpusCraft v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """CorpusCraft: Transform your documents into training datasets."""
    pass


@app.command()
def init(
    input_dir: Path = typer.Option(
        Path("./data"),
        "--input",
        "-i",
        help="Input directory with documents",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output",
        "-o",
        help="Output directory for datasets",
    ),
    config_file: Path = typer.Option(
        Path("./corpuscraft_config.yaml"),
        "--config",
        "-c",
        help="Path to save configuration file",
    ),
) -> None:
    """Initialize a new CorpusCraft project with default configuration."""
    console.print("\n[bold blue]Initializing CorpusCraft project...[/bold blue]\n")

    # Create default config
    config = CorpusCraftConfig(
        input=InputConfig(folder=input_dir),
        processing=ProcessingConfig(),
        llm=LLMConfig(backend=LLMBackend.OLLAMA, model="llama3.1:8b"),
        generators=[
            GeneratorConfig(
                type=GeneratorType.QA,
                qa=QAGeneratorConfig(num_examples=100),
            )
        ],
        output=OutputConfig(output_dir=output_dir),
    )

    # Save config
    config_dict = config.model_dump(mode="json")
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Created configuration file: {config_file}")
    console.print(f"[green]✓[/green] Input directory: {input_dir}")
    console.print(f"[green]✓[/green] Output directory: {output_dir}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Place your documents in the input directory")
    console.print("2. Edit the configuration file if needed")
    console.print(f"3. Run: corpuscraft generate -c {config_file}")
    console.print()


@app.command()
def generate(
    config_file: Path = typer.Option(
        Path("./corpuscraft_config.yaml"),
        "--config",
        "-c",
        help="Configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Generate dataset from documents using the configuration."""
    if verbose:
        logger.setLevel(logging.DEBUG)

    console.print("\n[bold blue]Starting CorpusCraft dataset generation...[/bold blue]\n")

    # Load config
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
        console.print("Run 'corpuscraft init' to create a default configuration.")
        raise typer.Exit(1)

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    config = CorpusCraftConfig(**config_dict)

    console.print(f"[green]✓[/green] Loaded configuration from {config_file}")

    # Step 1: Parse documents
    console.print("\n[bold]Step 1: Parsing documents[/bold]")

    parser = DoclingParser(config.processing)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing documents...", total=None)

        documents = parser.parse_directory(
            directory=config.input.folder,
            file_types=config.input.file_types,
            recursive=config.input.recursive,
        )

        progress.update(task, completed=True)

    stats = parser.get_statistics(documents)
    console.print(f"[green]✓[/green] Parsed {stats['total_documents']} documents")
    console.print(f"  Total chunks: {stats['total_chunks']}")
    console.print(f"  File types: {stats['file_types']}")

    # Step 2: Initialize LLM
    console.print("\n[bold]Step 2: Initializing LLM backend[/bold]")

    if config.llm.backend == LLMBackend.OLLAMA:
        llm = OllamaLLM(
            model=config.llm.model,
            base_url=config.llm.base_url or "http://localhost:11434",
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    else:
        console.print(f"[red]Error: Backend {config.llm.backend} not yet implemented[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Initialized {config.llm.backend} with {config.llm.model}")

    # Step 3: Generate dataset
    console.print("\n[bold]Step 3: Generating dataset[/bold]")

    all_examples = []

    for gen_config in config.generators:
        console.print(f"\n  Generator: {gen_config.type}")

        if gen_config.type == GeneratorType.QA:
            if not gen_config.qa:
                console.print("[yellow]Warning: No QA config provided, using defaults[/yellow]")
                gen_config.qa = QAGeneratorConfig()

            generator = QAGenerator(
                llm=llm,
                question_types=gen_config.qa.question_types,
                difficulty_levels=gen_config.qa.difficulty_levels,
                min_answer_length=gen_config.qa.min_answer_length,
                max_answer_length=gen_config.qa.max_answer_length,
            )

            examples = generator.generate(documents, gen_config.qa.num_examples)
            all_examples.extend(examples)

            console.print(f"  [green]✓[/green] Generated {len(examples)} QA pairs")

        else:
            console.print(f"  [yellow]Skipping unsupported generator: {gen_config.type}[/yellow]")

    # Step 4: Write output
    console.print("\n[bold]Step 4: Writing output[/bold]")

    writer = JSONLWriter(
        output_dir=config.output.output_dir,
        split_ratio=config.output.split_ratio,
        shuffle=config.output.shuffle,
        seed=config.output.seed,
    )

    output_files = writer.write(all_examples, dataset_name="dataset")

    console.print(f"[green]✓[/green] Wrote dataset to {config.output.output_dir}")

    # Show summary
    console.print("\n[bold green]Generation complete![/bold green]\n")

    table = Table(title="Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", justify="right", style="magenta")
    table.add_column("File", style="green")

    for split_name, file_path in output_files.items():
        num_examples = len(JSONLWriter.read_jsonl(file_path))
        table.add_row(split_name.capitalize(), str(num_examples), str(file_path.name))

    console.print(table)
    console.print()


@app.command()
def info(
    config_file: Path = typer.Argument(
        ...,
        help="Configuration file to inspect",
    ),
) -> None:
    """Display information about a configuration file."""
    if not config_file.exists():
        console.print(f"[red]Error: File not found: {config_file}[/red]")
        raise typer.Exit(1)

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    config = CorpusCraftConfig(**config_dict)

    console.print(f"\n[bold]Configuration: {config_file}[/bold]\n")
    console.print(f"Input folder: {config.input.folder}")
    console.print(f"File types: {', '.join(config.input.file_types)}")
    console.print(f"LLM backend: {config.llm.backend}")
    console.print(f"Model: {config.llm.model}")
    console.print(f"Output: {config.output.output_dir}")
    console.print(f"Generators: {len(config.generators)}")

    for i, gen in enumerate(config.generators, 1):
        console.print(f"  {i}. {gen.type}")

    console.print()


if __name__ == "__main__":
    app()

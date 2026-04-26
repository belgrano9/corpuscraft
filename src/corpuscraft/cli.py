from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="corpuscraft",
    help="Transform documents into ML training datasets.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    input_dir: Annotated[Path, typer.Option("--input", "-i", help="Folder with documents")] = Path("./data"),
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output folder")] = Path("./outputs"),
    config_file: Annotated[Path, typer.Option("--config", "-c", help="Config file path")] = Path("corpuscraft_config.yaml"),
) -> None:
    """Create a default configuration file."""
    from corpuscraft.config import save_default_config

    if config_file.exists():
        console.print(f"[yellow]Config already exists:[/] {config_file}")
        raise typer.Exit(1)

    save_default_config(config_file, input_dir, output_dir)
    console.print(f"[green]✓[/] Created config: [bold]{config_file}[/]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Place documents in [bold]{input_dir}[/]")
    console.print(f"  2. Edit [bold]{config_file}[/] to adjust pipeline and LLM settings")
    console.print(f"  3. Run: [bold]corpuscraft parse --config {config_file}[/]")
    console.print(f"  4. Run: [bold]corpuscraft generate --config {config_file}[/]")


@app.command()
def parse(
    config_file: Annotated[Path, typer.Option("--config", "-c", help="Config file")] = Path("corpuscraft_config.yaml"),
    input_dir: Annotated[Optional[Path], typer.Option("--input", "-i", help="Override input dir")] = None,
) -> None:
    """Parse documents and write Markdown files to outputs/parsed/."""
    from corpuscraft.config import load_config
    from corpuscraft.parsers import create_parser

    if not config_file.exists():
        console.print(f"[red]Config not found:[/] {config_file}")
        console.print("Run [bold]corpuscraft init[/] first.")
        raise typer.Exit(1)

    cfg = load_config(config_file)
    source = input_dir or cfg.input_dir

    if not source.exists():
        console.print(f"[red]Input directory not found:[/] {source}")
        raise typer.Exit(1)

    out_dir = cfg.exporter.output_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = create_parser(cfg.parser)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task(f"Scanning {source} ...", total=None)
        from corpuscraft.parsers.base import SUPPORTED_EXTENSIONS
        files = [
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        progress.update(task, description=f"Found {len(files)} document(s)")

    if not files:
        console.print(f"[yellow]No supported documents found in {source}[/]")
        raise typer.Exit()

    table = Table("File", "Status", "Size")
    for path in sorted(files):
        try:
            doc = parser.parse_file(path)
            out_path = out_dir / (path.stem + ".md")
            out_path.write_text(doc.content, encoding="utf-8")
            table.add_row(path.name, "[green]OK[/]", f"{len(doc):,} chars")
        except Exception as e:
            table.add_row(path.name, f"[red]FAILED[/]", str(e))

    console.print(table)
    console.print(f"\n[green]✓[/] Parsed files written to [bold]{out_dir}[/]")


@app.command()
def generate(
    config_file: Annotated[Path, typer.Option("--config", "-c", help="Config file")] = Path("corpuscraft_config.yaml"),
) -> None:
    """Run the full pipeline: parse → generate → export."""
    from corpuscraft.config import load_config
    from corpuscraft.exporters import export_jsonl
    from corpuscraft.generators.qa import QAGenerator
    from corpuscraft.parsers import create_parser
    from corpuscraft.parsers.base import SUPPORTED_EXTENSIONS

    if not config_file.exists():
        console.print(f"[red]Config not found:[/] {config_file}")
        raise typer.Exit(1)

    cfg = load_config(config_file)

    if not cfg.generators:
        console.print("[yellow]No generators configured in config file.[/]")
        raise typer.Exit()

    console.print(f"[bold]CorpusCraft[/] — pipeline: [cyan]{cfg.parser.pipeline}[/], LLM: [cyan]{cfg.llm.model}[/]")

    # 1. Parse
    parser = create_parser(cfg.parser)
    files = [
        p for p in cfg.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        console.print(f"[yellow]No documents found in {cfg.input_dir}[/]")
        raise typer.Exit()

    console.print(f"\nParsing {len(files)} document(s)...")
    documents = []
    for path in sorted(files):
        try:
            documents.append(parser.parse_file(path))
            console.print(f"  [green]✓[/] {path.name}")
        except Exception as e:
            console.print(f"  [red]✗[/] {path.name}: {e}")

    if not documents:
        console.print("[red]No documents parsed successfully.[/]")
        raise typer.Exit(1)

    # 2. Generate
    all_examples = []
    for gen_cfg in cfg.generators:
        if gen_cfg.type == "qa":
            generator = QAGenerator(gen_cfg, cfg.llm)
        else:
            console.print(f"[yellow]Unknown generator type '{gen_cfg.type}', skipping.[/]")
            continue

        console.print(f"\nGenerating {gen_cfg.num_examples} {gen_cfg.type.upper()} examples...")
        for doc in documents:
            examples = generator.generate(doc)
            all_examples.extend(examples)
            console.print(f"  [green]✓[/] {doc.source_path.name}: {len(examples)} examples")

    if not all_examples:
        console.print("[red]No examples generated.[/]")
        raise typer.Exit(1)

    # 3. Export
    written = export_jsonl(all_examples, cfg.exporter)
    console.print(f"\n[green]✓[/] Exported {len(all_examples)} examples:")
    for split, path in written.items():
        console.print(f"  {split}: [bold]{path}[/]")


if __name__ == "__main__":
    app()

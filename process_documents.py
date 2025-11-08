"""
Process all documents from data/raw folder and convert them to markdown.

This script:
1. Reads all supported documents from data/raw
2. Parses them using DoclingParser with Ollama VLM backend
3. Saves each document as a markdown file in data/processed

The markdown files preserve:
- Document structure and layout
- Mathematical formulas (LaTeX notation)
- Tables and formatting
- Text content with proper hierarchy
"""

from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from corpuscraft.parsers import DoclingParser

# Set up rich console
console = Console()

# Configuration
RAW_FOLDER = Path(__file__).parent / "data" / "raw"
PROCESSED_FOLDER = Path(__file__).parent / "data" / "processed"


def process_documents():
    """Process all documents from raw folder to markdown in processed folder."""

    # Ensure folders exist
    if not RAW_FOLDER.exists():
        console.print(f"[red]Raw folder not found: {RAW_FOLDER}[/red]")
        console.print("[yellow]Creating data/raw folder...[/yellow]")
        RAW_FOLDER.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created![/green] Please add documents to {RAW_FOLDER} and run again.")
        return

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing folder ready: {PROCESSED_FOLDER}")

    # Initialize parser with Ollama VLM backend (default)
    console.print("\n[bold cyan]═" * 35 + "[/bold cyan]")
    console.print("[bold]Initializing Docling Parser with Ollama VLM[/bold]")
    console.print("[bold cyan]═" * 35 + "[/bold cyan]\n")

    parser = DoclingParser()  # Uses Ollama by default now

    console.print(f"[dim]Supported formats: {', '.join(parser.get_supported_formats())}[/dim]\n")

    # Parse all documents in raw folder
    console.print(f"[bold]Scanning for documents in:[/bold] [cyan]{RAW_FOLDER}[/cyan]")

    documents = parser.parse_folder(RAW_FOLDER, recursive=True)

    if not documents:
        console.print("[yellow]No documents found to process![/yellow]")
        console.print(f"\nAdd documents to [cyan]{RAW_FOLDER}[/cyan] to process them.")
        console.print(f"Supported formats: {', '.join(parser.get_supported_formats())}")
        return

    console.print(f"\n[green]Found {len(documents)} documents to process[/green]\n")

    # Process each document with rich progress bar
    success_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("[cyan]Processing documents...", total=len(documents))

        for doc in documents:
            try:
                # Get the original file path and create output path
                original_path = doc.file_path

                # Preserve directory structure relative to raw folder
                relative_path = original_path.relative_to(RAW_FOLDER)

                # Create output path with .md extension
                output_path = PROCESSED_FOLDER / relative_path.with_suffix(".md")

                # Create subdirectories if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Update progress description
                progress.update(task, description=f"[cyan]Processing: {relative_path.name}")

                logger.info(
                    f"Processing {relative_path.name}: "
                    f"{len(doc.content):,} chars, "
                    f"{doc.metadata.get('file_size', 0):,} bytes"
                )

                # Write markdown content to file
                with open(output_path, "w", encoding="utf-8") as f:
                    # Add metadata header
                    f.write(f"<!-- Source: {relative_path} -->\n")
                    f.write(f"<!-- Format: {doc.file_format} -->\n")
                    if doc.page_count:
                        f.write(f"<!-- Pages: {doc.page_count} -->\n")
                    f.write(f"<!-- Processed with CorpusCraft + Docling + Ollama VLM -->\n\n")

                    # Write the parsed content
                    f.write(doc.content)

                logger.success(f"Saved: {output_path.relative_to(PROCESSED_FOLDER)}")
                success_count += 1

            except Exception as e:
                logger.error(f"Error processing {doc.file_path.name}: {e}")
                error_count += 1

            progress.update(task, advance=1)

    # Summary
    console.print(f"\n[bold green]═" * 35 + "[/bold green]")
    console.print("[bold green]Processing Complete![/bold green]")
    console.print(f"[bold green]═" * 35 + "[/bold green]")
    console.print(f"  [cyan]Total documents:[/cyan] {len(documents)}")
    console.print(f"  [green]Successfully processed:[/green] {success_count}")
    if error_count > 0:
        console.print(f"  [red]Errors:[/red] {error_count}")
    console.print(f"  [cyan]Output folder:[/cyan] {PROCESSED_FOLDER}")
    console.print(f"[bold green]═" * 35 + "[/bold green]\n")


def main():
    """Main entry point."""
    try:
        process_documents()
    except KeyboardInterrupt:
        logger.info("\n\nProcessing interrupted by user.")
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

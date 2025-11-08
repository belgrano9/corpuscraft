"""
Simple test script to verify the Docling parser works correctly.

This script demonstrates how to:
1. Initialize the DoclingParser with default settings (Ollama VLM by default)
2. Customize Ollama VLM backend settings
3. Parse documents from the data folder
4. Access parsed content and metadata
"""

from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from corpuscraft.parsers import DoclingParser

# Set up rich console for beautiful output
console = Console()

# Configuration: Customize Ollama settings if needed
USE_CUSTOM_OLLAMA_SETTINGS = False  # Change to True to customize


def main():
    """Test the Docling parser with documents from the data folder."""
    # Initialize the parser
    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print("[bold]Testing Docling Parser[/bold]")
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")

    if USE_CUSTOM_OLLAMA_SETTINGS:
        console.print("[yellow]Using custom Ollama VLM settings[/yellow]\n")
        parser = DoclingParser(
            ollama_url="http://localhost:11434/v1/chat/completions",
            ollama_model="gabegoodhart/granite-docling:258M",
            ocr_engine="tesseract",  # or "easyocr"
            vlm_timeout=300,
            vlm_scale=2.0,
        )
    else:
        console.print("[green]Using default Ollama VLM backend[/green]\n")
        parser = DoclingParser()  # Ollama enabled by default

    console.print(f"[dim]Supported formats: {', '.join(parser.get_supported_formats())}[/dim]\n")

    # Define the data folder path
    data_folder = Path(__file__).parent / "data"

    if not data_folder.exists():
        console.print(f"[red]Data folder not found: {data_folder}[/red]")
        return

    # Parse all documents in the data folder
    console.print(f"[bold]Parsing documents from:[/bold] [cyan]{data_folder}[/cyan]\n")
    documents = parser.parse_folder(data_folder, recursive=True)

    # Display results
    if not documents:
        console.print("[yellow]No documents were parsed![/yellow]")
        console.print(
            "\n[dim]To test the parser, add some documents to the data folder.[/dim]"
        )
        console.print("[dim]Supported formats: PDF, DOCX, PPTX, HTML, MD, TXT, PNG, JPG[/dim]")
        return

    console.print(f"\n[bold green]✓ Successfully parsed {len(documents)} documents[/bold green]\n")

    # Create a beautiful table with document information
    table = Table(title="Parsed Documents", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Format", style="green", width=8)
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("Pages", justify="right", style="blue", width=6)
    table.add_column("Characters", justify="right", style="magenta")
    table.add_column("Preview", style="dim", max_width=40)

    # Add rows for each document
    for i, doc in enumerate(documents, 1):
        file_name = doc.file_path.name if doc.file_path else "Unknown"
        file_format = doc.file_format or "?"
        file_size = f"{doc.metadata.get('file_size', 0):,} B"
        page_count = str(doc.page_count) if doc.page_count else "-"
        char_count = f"{len(doc.content):,}"

        # Show content preview (first 100 characters)
        preview = doc.content[:100].replace("\n", " ")
        if len(doc.content) > 100:
            preview += "..."

        table.add_row(
            str(i),
            file_name,
            file_format,
            file_size,
            page_count,
            char_count,
            preview,
        )

    console.print(table)

    console.print("\n[bold green]" + "═" * 60 + "[/bold green]")
    console.print("[bold green]Parser test completed successfully![/bold green]")
    console.print("[bold green]" + "═" * 60 + "[/bold green]\n")


if __name__ == "__main__":
    main()

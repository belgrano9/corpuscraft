import argparse
from collections import Counter
from pathlib import Path

from docling.document_converter import DocumentConverter


def main():
    parser = argparse.ArgumentParser(description="Trace and count Docling detections.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/Results3.pptx",
        help="Path to the document to parse (PDF, PPTX, DOCX, etc.)"
    )
    args = parser.add_argument(
        "--show-items",
        action="store_true",
        help="Print each detected item individually."
    )
    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Parsing document: {file_path.name} ...")
    
    # Initialize DocumentConverter without special options to see default detections
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    doc = result.document

    print(f"\nSuccessfully parsed! Document format: {file_path.suffix}")
    print(f"Total Pages/Slides: {len(doc.pages)}")
    print("-" * 50)

    # Dictionary to keep track of detection types and their counts
    detections = Counter()
    text_counts = Counter()

    # We iterate over all elements in reading order
    for item, level in doc.iterate_items():
        # Get the label/type of the item
        label = item.label.value if hasattr(item, "label") else type(item).__name__
        detections[label] += 1
        
        if args.show_items:
            # Safely get text if available
            text_preview = item.text[:50].replace('\n', ' ') if hasattr(item, "text") else ""
            print(f"[{label}] (level {level}): {text_preview}")

        if hasattr(item, "text") and item.text:
            cleaned_text = item.text.strip()
            if cleaned_text:
                text_counts[cleaned_text] += 1

    # Print the summary report
    print("DETECTION SUMMARY:")
    if not detections:
        print("  No elements detected.")
    else:
        for label, count in detections.most_common():
            print(f"  - {label.ljust(20)}: {count}")
    print("-" * 50)

    # Print duplicates report
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    if duplicates:
        print("\nDUPLICATE TEXTS DETECTED:")
        for text, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            preview = text[:60].replace('\n', ' ') + ("..." if len(text) > 60 else "")
            print(f"  - [{count}x] {preview}")
        print("-" * 50)


if __name__ == "__main__":
    main()

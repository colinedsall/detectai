#!/usr/bin/env python3
"""
PDF Importer for DetectAI Training Data.
Extracts text from PDF files for use as human-written training samples.
"""
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)


class PDFImporter:
    """Import and extract text from PDF files."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the PDF importer."""
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, 'training_data', 'human')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract all text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            return full_text if full_text.strip() else None
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Get metadata about a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            info = {
                'filename': os.path.basename(pdf_path),
                'pages': len(doc),
                'metadata': doc.metadata,
            }
            doc.close()
            return info
        except Exception as e:
            return {'filename': os.path.basename(pdf_path), 'error': str(e)}
    
    def import_pdf(self, pdf_path: str, label: str = "human") -> Optional[Dict]:
        """
        Import a PDF file and save as training data.
        
        Args:
            pdf_path: Path to the PDF file
            label: Label for the text (human or ai)
            
        Returns:
            Dictionary with import results or None if failed
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return None
        
        text = self.extract_text(pdf_path)
        if not text:
            print(f"No text extracted from: {pdf_path}")
            return None
        
        # Get word count
        word_count = len(text.split())
        if word_count < 100:
            print(f"Skipping {pdf_path}: too short ({word_count} words)")
            return None
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
        output_filename = f"pdf_{safe_name}.txt"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {pdf_path}\n")
            f.write(f"Type: PDF Import\n")
            f.write(f"Word Count: {word_count}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("-" * 80 + "\n\n")
            f.write(text)
        
        result = {
            'source': pdf_path,
            'output': output_path,
            'word_count': word_count,
            'label': label
        }
        
        print(f" Imported: {base_name} ({word_count} words)")
        return result
    
    def import_directory(self, directory: str, label: str = "human") -> List[Dict]:
        """Import all PDF files from a directory."""
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            return []
        
        results = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            result = self.import_pdf(pdf_path, label)
            if result:
                results.append(result)
        
        return results


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import PDF files as training data")
    parser.add_argument("path", help="PDF file or directory to import")
    parser.add_argument("--label", default="human", choices=["human", "ai"],
                        help="Label for the imported text")
    parser.add_argument("--output", help="Output directory for extracted text")
    
    args = parser.parse_args()
    
    importer = PDFImporter(output_dir=args.output)
    
    if os.path.isdir(args.path):
        results = importer.import_directory(args.path, args.label)
        print(f"\n Imported {len(results)} PDF files")
    else:
        result = importer.import_pdf(args.path, args.label)
        if result:
            print(f"\n Import successful: {result['output']}")
        else:
            print("\n Import failed")


if __name__ == "__main__":
    main()

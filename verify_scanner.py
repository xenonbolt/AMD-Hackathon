import sys
from unittest.mock import MagicMock

# Mock torch, transformers, and peft to allow importing scanner/inference engine without dependencies
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()

from scanner import extract_java_blocks

def test_scanner_extraction():
    mock_java_code = """
    package com.example;
    
    public class SecurityTester {
        // Line comment with { brace
        /* Block comment with } brace */
        
        public void vulnerableMethod(String input) {
            String sql = "SELECT * FROM users WHERE id = " + input; // { inside string
            System.out.println("Processing...");
        }
        
        private int secureMethod(int val) {
            return val * 2;
        }
    }
    """
    
    print("Testing Java block extraction...")
    chunks = extract_java_blocks(mock_java_code)
    
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} (Lines {chunk['start_line']} to {chunk['end_line']}) ---")
        print(chunk["content"].strip())
        
    # Check if we got 2 method chunks
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    print("\nExtraction test completed successfully!")

if __name__ == "__main__":
    try:
        test_scanner_extraction()
    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        sys.exit(1)

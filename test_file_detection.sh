#!/bin/bash

echo "🤖 Testing AI-Generated Text File Detection"
echo "=========================================="

echo -e "\n1. Testing with the AI-generated text file:"
curl -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "sample_ai_text.txt"}' | jq '.'

echo -e "\n2. Testing with direct text input (for comparison):"
curl -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a simple human-written sentence for comparison."}' | jq '.'

echo -e "\n3. Quick comparison - just the AI probabilities:"
echo "AI Text File:"
curl -s -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "sample_ai_text.txt"}' | jq -r '.probability_ai'

echo "Human Text:"
curl -s -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a simple human-written sentence for comparison."}' | jq -r '.probability_ai'

echo -e "\n✅ File detection test completed!"

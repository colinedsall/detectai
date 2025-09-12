#!/bin/bash

#  **ADD YOUR FILE PATHS HERE** 
FILES=(
    "sample_ai_text.txt"                    # Example from our test
    # "/Users/colin/Documents/my_essay.txt"    # Your essay
    # "/Users/colin/Documents/chatgpt.txt"     # ChatGPT response
    # "/Users/colin/Documents/human.txt"       # Human-written text
    # "~/Documents/suspicious.txt"             # Suspicious content
    # Add as many as you want!
)

echo " AI Text Detection - Batch Analysis"
echo "======================================"
echo " Files to analyze: ${#FILES[@]}"
echo

# Check which files exist
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo " $file"
    else
        echo " $file (not found)"
    fi
done

echo
echo " Starting analysis..."
echo

# Analyze each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo " Analyzing: $file"
        echo "----------------------------------------"
        
        # Make API request
        response=$(curl -s -X POST "http://127.0.0.1:8000/v1/detect/text" \
          -H "Content-Type: application/json" \
          -d "{\"file_path\": \"$file\"}")
        
        if [ $? -eq 0 ]; then
            # Extract key metrics
            ai_prob=$(echo "$response" | jq -r '.probability_ai // 0')
            confidence=$(echo "$response" | jq -r '.confidence // 0')
            word_count=$(echo "$response" | jq -r '.metadata.word_count // 0')
            unique_words=$(echo "$response" | jq -r '.metadata.unique_words // 0')
            highlight_count=$(echo "$response" | jq -r '.highlight_spans | length // 0')
            
            # Format percentages
            ai_prob_pct=$(printf "%.1f%%" $(echo "$ai_prob * 100" | bc -l))
            confidence_pct=$(printf "%.1f%%" $(echo "$confidence * 100" | bc -l))
            diversity_pct=$(printf "%.1f%%" $(echo "$unique_words * 100 / $word_count" | bc -l))
            
            echo " AI Probability: $ai_prob_pct"
            echo " Confidence: $confidence_pct"
            echo " Stats: $word_count words, $unique_words unique ($diversity_pct diversity)"
            echo " Suspicious Sections: $highlight_count"
            
            # Assessment
            if (( $(echo "$ai_prob > 0.7" | bc -l) )); then
                echo " Assessment:  HIGH likelihood of AI-generated content"
            elif (( $(echo "$ai_prob > 0.5" | bc -l) )); then
                echo " Assessment:  MODERATE likelihood of AI-generated content"
            else
                echo " Assessment:  LOW likelihood of AI-generated content"
            fi
        else
            echo " Failed to analyze file"
        fi
        
        echo
    fi
done

echo " Batch analysis completed!"
echo
echo " To add more files, edit the FILES array in this script."

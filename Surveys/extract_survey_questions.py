#!/usr/bin/env python3
"""
Extract survey questions from PRE-TASK SURVEY.md and save as structured JSON.

This script parses the markdown survey file and extracts:
- Section metadata (name, reference, scale information)
- Questions with section prefixes
- Outputs structured JSON for use by survey_response_builder.py
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def extract_survey_questions(md_path: str, output_json_path: str) -> Dict[str, Any]:
    """
    Extract survey questions from markdown file.
    
    Args:
        md_path: Path to PRE-TASK SURVEY.md
        output_json_path: Path to save survey_questions.json
    
    Returns:
        Dictionary with survey structure
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    survey_data = {
        "survey_metadata": {
            "survey_name": "PRE-TASK SURVEY",
            "survey_version": "1.0",
            "extraction_date": datetime.now().isoformat()
        },
        "sections": []
    }
    
    current_section = None
    current_questions = []
    current_prefix = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for section header (### [**Section X: ...**])
        section_match = re.match(r'###\s*\[\*\*Section\s+\d+:\s*(.+?)\*\*\]', line)
        if section_match:
            # Save previous section if exists
            if current_section and current_questions:
                current_section["questions"] = current_questions
                survey_data["sections"].append(current_section)
            
            # Start new section
            section_name = section_match.group(1)
            section_id = extract_section_id(section_name)
            
            # Get reference (next line after section header)
            reference = ""
            if i + 1 < len(lines):
                ref_line = lines[i + 1].strip()
                if ref_line.startswith('*') and ref_line.endswith('*'):
                    reference = ref_line.strip('*')
                    i += 1
            
            # Get scale information
            scale_info = None
            scale_labels = None
            if i + 1 < len(lines):
                scale_line = lines[i + 1].strip()
                if scale_line.startswith('**Scale:**'):
                    scale_info = parse_scale(scale_line)
                    scale_labels = scale_info["labels"]
                    i += 1
            
            current_section = {
                "section_id": section_id,
                "section_name": section_name,
                "reference": reference,
                "scale": {
                    "min": scale_info["min"] if scale_info else 1,
                    "max": scale_info["max"] if scale_info else 5,
                    "labels": scale_labels if scale_labels else {}
                }
            }
            current_questions = []
            current_prefix = None
            i += 1
            continue
        
        # Check for section prefix (e.g., "I see myself as someone who…")
        if current_section and not current_prefix:
            # Look for lines that end with "…" or contain "who" - likely a prefix
            if line and (line.endswith('…') or ('who' in line.lower() and len(line) < 100)):
                current_prefix = line
                i += 1
                continue
        
        # Check for numbered question (e.g., "1. is reserved.")
        question_match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if question_match and current_section:
            question_num = int(question_match.group(1))
            question_text = question_match.group(2).strip()
            
            # If there's a prefix, prepend it to the question
            full_question = question_text
            if current_prefix:
                # Remove trailing "…" and add proper punctuation
                prefix = current_prefix.rstrip('…').strip()
                if not prefix.endswith(' '):
                    prefix += ' '
                full_question = prefix + question_text
            
            question_id = f"{current_section['section_id']}_{question_num}"
            
            current_questions.append({
                "question_id": question_id,
                "question_text": full_question,
                "section_prefix": current_prefix if current_prefix else None
            })
            i += 1
            continue
        
        # Skip empty lines and other content
        i += 1
    
    # Save last section
    if current_section and current_questions:
        current_section["questions"] = current_questions
        survey_data["sections"].append(current_section)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(survey_data, f, indent=2, ensure_ascii=False)
    
    total_questions = sum(len(s["questions"]) for s in survey_data["sections"])
    print(f"Extracted {len(survey_data['sections'])} sections with {total_questions} total questions")
    print(f"Saved to: {output_json_path}")
    
    return survey_data


def extract_section_id(section_name: str) -> str:
    """Extract section ID from section name."""
    # BFI-10, BES-A, REI patterns
    if 'BFI-10' in section_name or 'Big Five' in section_name:
        return "BFI-10"
    elif 'BES-A' in section_name or 'Empathy Scale' in section_name:
        return "BES-A"
    elif 'REI' in section_name or 'Rational-Experiential' in section_name:
        return "REI"
    else:
        # Fallback: use first few words
        words = section_name.split()
        return "_".join(words[:3]).upper().replace('(', '').replace(')', '').replace(',', '')


def parse_scale(scale_line: str) -> Dict[str, Any]:
    """Parse scale information from scale line."""
    # Format: **Scale:** 1 \= Disagree Strongly   2 \= Disagree a Little ...
    scale_line = scale_line.replace('**Scale:**', '').strip()
    
    # Extract min and max
    numbers = re.findall(r'\d+', scale_line)
    min_val = int(numbers[0]) if numbers else 1
    max_val = int(numbers[-1]) if len(numbers) > 1 else int(numbers[0]) if numbers else 5
    
    # Extract labels - handle escaped equals signs
    labels = {}
    # Pattern: "1 \= Disagree Strongly" or "1 = Disagree Strongly"
    # Split by number followed by equals (with optional backslash)
    parts = re.split(r'\s+(\d+)\s*\\?=\s*', scale_line)
    
    # parts will be: [prefix, '1', 'label1', '2', 'label2', ...]
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                num_str = parts[i]
                # Get label (everything until next number or end)
                label = parts[i + 1].strip()
                # Remove trailing whitespace and clean up
                label = re.sub(r'\s+', ' ', label).strip()
                labels[num_str] = label
    
    # Fallback: try simpler pattern matching
    if not labels:
        # Match "N \= Label" patterns
        pattern = r'(\d+)\s*\\?=\s*([A-Za-z\s]+?)(?=\s+\d+\s*\\?=|$)'
        matches = re.findall(pattern, scale_line)
        for num_str, label in matches:
            labels[num_str] = label.strip()
    
    return {
        "min": min_val,
        "max": max_val,
        "labels": labels
    }


def main():
    script_dir = Path(__file__).parent
    md_path = script_dir / "PRE-TASK SURVEY.md"
    output_path = script_dir / "survey_questions.json"
    
    if not md_path.exists():
        print(f"Error: Survey markdown file not found at {md_path}")
        return
    
    extract_survey_questions(str(md_path), str(output_path))


if __name__ == "__main__":
    main()


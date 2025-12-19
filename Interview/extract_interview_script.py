#!/usr/bin/env python3
"""
Extract interview questions from Interview_Script.md.

The script parses the standardized interview script where each question is followed
by a number on the next line representing estimated_duration in seconds.

Format:
[Question text]
[Duration in seconds]

Note: This extracts the standardized script, not personalized follow-up questions.
For future enhancement: The paper mentions an AI interviewer that generates dynamic
follow-up questions. This could be implemented later to make interviews more adaptive.
"""

import json
import os
import re
from pathlib import Path


def extract_interview_script(md_path, output_json_path):
    """
    Extract interview questions from Interview_Script.md file.
    
    Format: Questions can span multiple lines, followed by a duration number.
    Example:
        To start, I would like to begin with a big question: tell me the story of your life. Start from the beginning --
        from your childhood, to education, to family and relationships, and to any major life events you may have
        had.
        625
    
    Args:
        md_path: Path to the Interview_Script.md file
        output_json_path: Path where interview_script.json will be saved
    """
    with open(md_path, 'r') as f:
        lines = f.readlines()
    
    interview_script = []
    question_id = 1
    
    # Skip header lines: "[Interview Script]", "Script", "Time Limit (sec)"
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip header lines
        if line in ["[Interview Script]", "Script", "Time Limit (sec)", ""]:
            i += 1
            continue
        
        # Check if this line is a number (duration)
        if line.isdigit():
            # Collect previous non-empty lines as the question (may span multiple lines)
            question_parts = []
            j = i - 1
            while j >= 0:
                prev_line = lines[j].strip()
                if prev_line == "":
                    break
                if prev_line.isdigit():
                    break
                if prev_line in ["[Interview Script]", "Script", "Time Limit (sec)"]:
                    break
                question_parts.insert(0, prev_line)
                j -= 1
            
            if question_parts:
                question = " ".join(question_parts)
                duration = int(line)
                
                interview_script.append({
                    "question_id": question_id,
                    "question": question,
                    "estimated_duration": duration
                })
                question_id += 1
            i += 1
        else:
            # This is a question line, check if next line is duration
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.isdigit():
                    # Collect current and previous question lines
                    question_parts = [line]
                    j = i - 1
                    while j >= 0:
                        prev_line = lines[j].strip()
                        if prev_line == "" or prev_line.isdigit() or prev_line in ["[Interview Script]", "Script", "Time Limit (sec)"]:
                            break
                        question_parts.insert(0, prev_line)
                        j -= 1
                    
                    question = " ".join(question_parts)
                    duration = int(next_line)
                    
                    interview_script.append({
                        "question_id": question_id,
                        "question": question,
                        "estimated_duration": duration
                    })
                    question_id += 1
                    i += 2  # Skip both question and duration lines
                else:
                    # Continue collecting question lines
                    i += 1
            else:
                # Last line, might be part of a question without duration
                i += 1
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(interview_script, f, indent=2)
    
    print(f"Extracted {len(interview_script)} interview questions")
    print(f"Saved to: {output_json_path}")
    
    return interview_script


if __name__ == "__main__":
    # Get project root (assuming script is in agent_bank/scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Paths
    interview_script_md = project_root / "Research Paper/Interview_Script.md"
    output_path = script_dir / "interview_script.json"
    
    if not interview_script_md.exists():
        print(f"Error: Interview script not found at {interview_script_md}")
        exit(1)
    
    extract_interview_script(str(interview_script_md), str(output_path))


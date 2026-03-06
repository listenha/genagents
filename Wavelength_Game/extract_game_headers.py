#!/usr/bin/env python3
"""
Extract game headers from Game Headers.csv and save as game_headers.json.

This script reads the CSV source of truth (Cue, Spectrum Left, Spectrum Right)
and produces the structured JSON consumed by WavelengthResponseBuilder and
run_wavelength.py.

Usage:
    python3 Wavelength_Game/extract_game_headers.py

Output: Wavelength_Game/game_headers.json
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def extract_game_headers(csv_path: str, output_json_path: str) -> List[Dict[str, Any]]:
    """
    Read game headers from CSV and write structured JSON.

    Args:
        csv_path: Path to Game Headers.csv (columns: Cue, Spectrum Left, Spectrum Right)
        output_json_path: Path to write game_headers.json

    Returns:
        List of header dicts with spectrum and cue
    """
    headers_list: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cue = (row.get("Cue") or "").strip()
            left = (row.get("Spectrum Left") or "").strip()
            right = (row.get("Spectrum Right") or "").strip()
            if not cue and not left and not right:
                continue
            if not cue or not left or not right:
                raise ValueError(
                    f"Row has missing field: Cue={repr(cue)}, Spectrum Left={repr(left)}, Spectrum Right={repr(right)}"
                )
            headers_list.append({
                "spectrum": {"left": left, "right": right},
                "cue": cue,
            })

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(headers_list, f, indent=2)

    return headers_list


def main() -> None:
    script_dir = Path(__file__).parent
    csv_path = script_dir / "Game Headers.csv"
    output_path = script_dir / "game_headers.json"

    if not csv_path.exists():
        print(f"Error: Game headers CSV not found at {csv_path}")
        return

    headers = extract_game_headers(str(csv_path), str(output_path))
    print(f"Extracted {len(headers)} game headers to {output_path}")


if __name__ == "__main__":
    main()

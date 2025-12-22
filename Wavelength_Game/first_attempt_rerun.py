# Wavelength_Game/first_attempt_rerun.py
#!/usr/bin/env python3
"""
Patch script to re-run first attempts for agents with null responses.
Only runs for agents that have null first attempts but valid second attempts.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Wavelength_Game.wavelength_response_builder import WavelengthResponseBuilder

def has_null_first_attempt(agent_folder):
    """Check if agent has null first attempt or is missing attempt_id 1."""
    responses_path = Path(agent_folder) / "wavelength_responses.json"
    if not responses_path.exists():
        return False
    
    with open(responses_path, 'r') as f:
        data = json.load(f)
    
    if not data.get("attempts"):
        return False
    
    # Check if attempt_id 1 exists
    attempt_1 = None
    for attempt in data["attempts"]:
        if attempt.get("attempt_id") == 1:
            attempt_1 = attempt
            break
    
    # Case 1: attempt_id 1 doesn't exist at all (was removed by previous failed run)
    if attempt_1 is None:
        # Check if there's a valid second attempt (attempt_id 2)
        has_valid_second = False
        for attempt in data["attempts"]:
            if attempt.get("attempt_id") == 2:
                headers = attempt.get("headers", {})
                has_valid_second = any(
                    h.get("response") is not None 
                    for h in headers.values()
                )
                break
        return has_valid_second
    
    # Case 2: attempt_id 1 exists but all responses are null
    headers = attempt_1.get("headers", {})
    all_null = all(
        h.get("response") is None 
        for h in headers.values()
    )
    
    # Also check if there's a valid second attempt
    has_valid_second = False
    for attempt in data["attempts"]:
        if attempt.get("attempt_id") == 2:
            second_headers = attempt.get("headers", {})
            has_valid_second = any(
                h.get("response") is not None 
                for h in second_headers.values()
            )
            break
    
    return all_null and has_valid_second

def main():
    base_path = Path("agent_bank/populations/gss_agents")
    game_headers_path = "Wavelength_Game/game_headers.json"
    
    # Find agents 0057-0099 with null first attempts
    agents_to_patch = []
    for agent_id in range(57, 100):
        agent_folder = base_path / f"{agent_id:04d}"
        if agent_folder.exists() and has_null_first_attempt(agent_folder):
            agents_to_patch.append(str(agent_folder))
    
    if not agents_to_patch:
        print("No agents found that need patching.")
        return
    
    print(f"Found {len(agents_to_patch)} agents with null first attempts:")
    for folder in agents_to_patch:
        print(f"  - {folder}")
    
    print(f"\nPatching first attempts...")
    
    # Initialize builder
    builder = WavelengthResponseBuilder(game_headers_path, config={
        "include_reasoning": True,
        "batch_by_header": False
    })
    
    # Re-run first attempt for each agent
    for agent_folder in agents_to_patch:
        print(f"\n{'='*60}")
        print(f"Patching: {agent_folder}")
        print(f"{'='*60}")
        
        try:
            # Load existing data
            responses_data = builder._load_agent_responses(agent_folder)
            
            # Store original attempts 2 and 3 (if they exist) for verification
            original_attempts_2_3 = []
            if len(responses_data["attempts"]) > 1:
                # Store attempts 2 and 3 (indices 1 and 2)
                for idx in range(1, min(3, len(responses_data["attempts"]))):
                    original_attempts_2_3.append(responses_data["attempts"][idx].copy())
            
            # Remove the null first attempt (if it exists)
            # Find and remove attempt_id 1 if it exists
            attempt_1_idx = None
            for idx, attempt in enumerate(responses_data["attempts"]):
                if attempt.get("attempt_id") == 1:
                    attempt_1_idx = idx
                    break
            
            if attempt_1_idx is not None:
                responses_data["attempts"].pop(attempt_1_idx)
            
            # Save the modified data (removing null first attempt)
            # This ensures attempts 2 and 3 are preserved
            builder._save_agent_responses(agent_folder, responses_data)
            
            # Now run a new first attempt
            result = builder.administer_game(agent_folder, attempt_id=1, include_reasoning=True)
            
            # Reload to get the new structure
            responses_data = builder._load_agent_responses(agent_folder)
            
            # Fix ordering: insert new attempt 1 at the beginning
            # Find the new attempt (should be at the end after append)
            new_attempt = None
            remaining_attempts = []
            for attempt in responses_data["attempts"]:
                if attempt.get("attempt_id") == 1:
                    new_attempt = attempt
                else:
                    remaining_attempts.append(attempt)
            
            if new_attempt:
                # Reconstruct with correct order: [new_attempt1, attempt2, attempt3]
                responses_data["attempts"] = [new_attempt] + remaining_attempts
                # Ensure attempt IDs are correct
                for idx, attempt in enumerate(responses_data["attempts"]):
                    attempt["attempt_id"] = idx + 1
                
                # Verify attempts 2 and 3 are unchanged
                if len(original_attempts_2_3) > 0:
                    for orig_idx, orig_attempt in enumerate(original_attempts_2_3):
                        new_idx = orig_idx + 1  # attempt 2 is at index 1, attempt 3 at index 2
                        if new_idx < len(responses_data["attempts"]):
                            new_attempt_data = responses_data["attempts"][new_idx]
                            # Compare key fields (headers should be identical)
                            if new_attempt_data.get("headers") != orig_attempt.get("headers"):
                                print(f"WARNING: Attempt {new_idx + 1} data may have changed!")
                
                builder._save_agent_responses(agent_folder, responses_data)
            
            print(f"✓ Patched: {result['headers_completed']}/{result['total_headers']} headers")
            print(f"✓ Attempts 2 and 3 preserved and unchanged")
            
        except Exception as e:
            print(f"ERROR patching {agent_folder}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("PATCHING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
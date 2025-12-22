"""
Wavelength Response Builder

A modular system to administer Wavelength games to agents and store responses in structured JSON format.
Game interactions are NOT recorded to memory stream - only responses are saved.

Features:
- Uses numerical_resp() method with Wavelength-specific prompt template
- Supports multiple attempts per agent
- Stores numerical responses (0-100) + optional reasoning
- Requires all headers to be completed
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch  # For GPU memory management

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation_engine.settings import *
from genagents.genagents import GenerativeAgent


class WavelengthResponseBuilder:
    """
    Builds Wavelength game responses for agents and stores them in structured JSON format.
    
    Game interactions are NOT saved to memory stream - only final responses are stored.
    """
    
    def __init__(self, game_headers_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the WavelengthResponseBuilder.
        
        Args:
            game_headers_path: Path to game_headers.json
            config: Optional configuration dictionary
        """
        self.game_headers_path = game_headers_path
        self.game_headers = self._load_game_headers()
        
        # Default configuration
        default_config = {
            "include_reasoning": True,  # Whether to store raw reasoning text
            "require_all_headers": True,  # Must complete all headers
            "batch_by_header": False,  # If True: send all headers at once. If False: one header per inference
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
    
    def _load_game_headers(self) -> List[Dict[str, Any]]:
        """Load game headers from JSON file."""
        if not os.path.exists(self.game_headers_path):
            raise FileNotFoundError(f"Game headers file not found: {self.game_headers_path}")
        
        with open(self.game_headers_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_agent_responses(self, agent_folder: str) -> Dict[str, Any]:
        """Load existing game responses for an agent."""
        responses_path = Path(agent_folder) / "wavelength_responses.json"
        
        if responses_path.exists():
            # Check if file is empty or invalid
            if os.path.getsize(responses_path) == 0:
                print(f"    Warning: {responses_path} is empty, creating new structure", flush=True)
                return {
                    "game_metadata": {
                        "game_name": "Wavelength Game",
                        "game_version": "1.0",
                        "extraction_date": datetime.now().isoformat()
                    },
                    "agent_metadata": {},
                    "attempts": []
                }
            
            try:
                with open(responses_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate structure
                    if not isinstance(data, dict) or "attempts" not in data:
                        print(f"    Warning: {responses_path} has invalid structure, creating new structure", flush=True)
                        return {
                            "game_metadata": {
                                "game_name": "Wavelength Game",
                                "game_version": "1.0",
                                "extraction_date": datetime.now().isoformat()
                            },
                            "agent_metadata": {},
                            "attempts": []
                        }
                    return data
            except json.JSONDecodeError as e:
                print(f"    Warning: {responses_path} contains invalid JSON: {e}", flush=True)
                print(f"    Creating new structure", flush=True)
                return {
                    "game_metadata": {
                        "game_name": "Wavelength Game",
                        "game_version": "1.0",
                        "extraction_date": datetime.now().isoformat()
                    },
                    "agent_metadata": {},
                    "attempts": []
                }
        else:
            # Return empty structure
            return {
                "game_metadata": {
                    "game_name": "Wavelength Game",
                    "game_version": "1.0",
                    "extraction_date": datetime.now().isoformat()
                },
                "agent_metadata": {},
                "attempts": []
            }
    
    def _save_agent_responses(self, agent_folder: str, responses_data: Dict[str, Any]):
        """Save game responses to agent folder."""
        responses_path = Path(agent_folder) / "wavelength_responses.json"
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(responses_path, 'w', encoding='utf-8') as f:
            json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    def _format_game_header_for_llm(self, header: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Format a game header for LLM numerical_resp() method.
        
        Args:
            header: Game header dict with spectrum and cue
        
        Returns:
            Dictionary mapping question text to [0, 100] range
        """
        spectrum_left = header["spectrum"]["left"]
        spectrum_right = header["spectrum"]["right"]
        cue = header["cue"]
        
        # Format as: "Spectrum: [LEFT] — [RIGHT]\nClue: [CUE]\nWhere on this spectrum (0-100) does the clue belong?"
        question_text = f"Spectrum: {spectrum_left} — {spectrum_right}\nClue: {cue}\nWhere on this spectrum (0-100) does the clue belong?"
        
        return {question_text: [0, 100]}
    
    def _administer_game(self, agent: GenerativeAgent) -> Dict[str, Any]:
        """
        Administer all game headers to an agent.
        
        Can operate in two modes:
        - batch_by_header=False: One header per inference (default, more reliable)
        - batch_by_header=True: All headers at once (faster, but may truncate)
        
        Returns:
            Dictionary with headers responses and optional raw_responses
        """
        num_headers = len(self.game_headers)
        batch_mode = self.config.get("batch_by_header", False)
        
        if batch_mode:
            return self._administer_game_batch(agent)
        else:
            return self._administer_game_single(agent)
    
    def _administer_game_single(self, agent: GenerativeAgent) -> Dict[str, Any]:
        """
        Administer game to an agent, one header at a time.
        
        Returns:
            Dictionary with header responses and optional raw_responses
        """
        num_headers = len(self.game_headers)
        print(f"    Administering game ({num_headers} headers, one at a time)...", flush=True)
        
        headers_responses = {}
        game_start_time = time.time()
        header_times = []
        
        # Process each header individually
        for i, header in enumerate(self.game_headers):
            header_id = f"header_{i}"
            spectrum_left = header["spectrum"]["left"]
            spectrum_right = header["spectrum"]["right"]
            cue = header["cue"]
            
            # Format header as question
            question_dict = self._format_game_header_for_llm(header)
            
            # Time the header
            header_start_time = time.time()
            print(f"      Header {i+1}/{num_headers} ({cue})...", end=" ", flush=True)
            
            try:
                # Get agent response using Wavelength-specific template
                # Use singular template for single header
                prompt_template = "wavelength_singular_v1.txt"
                response_data = agent.numerical_resp(question_dict, float_resp=False, prompt_template=prompt_template)
                
                # Validate response
                if not response_data or "responses" not in response_data or len(response_data["responses"]) == 0:
                    print(f"ERROR: No response received", flush=True)
                    headers_responses[header_id] = {
                        "spectrum": {"left": spectrum_left, "right": spectrum_right},
                        "cue": cue,
                        "response": None
                    }
                    if self.config["include_reasoning"]:
                        headers_responses[header_id]["raw_response"] = "ERROR: No response received"
                    continue
                
                # Extract response (should be single value, 0-100)
                response_value = response_data["responses"][0]
                # Validate range
                if response_value < 0 or response_value > 100:
                    print(f"WARNING: Response {response_value} out of range [0-100], clamping", flush=True)
                    response_value = max(0, min(100, response_value))
                
                headers_responses[header_id] = {
                    "spectrum": {"left": spectrum_left, "right": spectrum_right},
                    "cue": cue,
                    "response": int(response_value)
                }
                
                # Store reasoning if enabled
                if self.config["include_reasoning"]:
                    if len(response_data.get("reasonings", [])) > 0:
                        headers_responses[header_id]["raw_response"] = response_data["reasonings"][0]
                    else:
                        headers_responses[header_id]["raw_response"] = "No reasoning provided"
                
                header_time = time.time() - header_start_time
                header_times.append(header_time)
                print(f"✓ ({header_time:.2f}s, response: {response_value})", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Free up fragmented memory
                
            except (ValueError, TypeError, IndexError) as e:
                header_time = time.time() - header_start_time
                header_times.append(header_time)
                print(f"ERROR: {e} ({header_time:.2f}s)", flush=True)
                headers_responses[header_id] = {
                    "spectrum": {"left": spectrum_left, "right": spectrum_right},
                    "cue": cue,
                    "response": None
                }
                if self.config["include_reasoning"]:
                    headers_responses[header_id]["raw_response"] = f"ERROR: {str(e)}"
        
        game_time = time.time() - game_start_time
        avg_time = sum(header_times) / len(header_times) if header_times else 0
        print(f"    Game completed: {game_time:.2f}s total (avg {avg_time:.2f}s per header)", flush=True)
        
        return headers_responses
    
    def _administer_game_batch(self, agent: GenerativeAgent) -> Dict[str, Any]:
        """
        Administer game to an agent, all headers in one inference.
        
        Returns:
            Dictionary with header responses and optional raw_responses
        """
        num_headers = len(self.game_headers)
        print(f"    Administering game ({num_headers} headers, all at once)...", flush=True)
        
        headers_responses = {}
        game_start_time = time.time()
        
        # Format all headers as questions
        questions_dict = {}
        header_order = []  # Track order to map responses back
        
        for i, header in enumerate(self.game_headers):
            header_id = f"header_{i}"
            question_dict = self._format_game_header_for_llm(header)
            questions_dict.update(question_dict)
            header_order.append((header_id, header))
        
        # Send all headers at once
        print(f"      Sending {num_headers} headers in one inference...", end=" ", flush=True)
        
        try:
            # Get agent response using Wavelength-specific batch template
            prompt_template = "wavelength_batch_v1.txt"
            response_data = agent.numerical_resp(questions_dict, float_resp=False, prompt_template=prompt_template)
            
            # Validate response
            if not response_data or "responses" not in response_data:
                print(f"ERROR: No response received", flush=True)
                # Set all responses to None
                for header_id, header in header_order:
                    headers_responses[header_id] = {
                        "spectrum": {"left": header["spectrum"]["left"], "right": header["spectrum"]["right"]},
                        "cue": header["cue"],
                        "response": None
                    }
                    if self.config["include_reasoning"]:
                        headers_responses[header_id]["raw_response"] = "ERROR: No response received"
            else:
                # Map responses back to header_ids
                response_values = response_data["responses"]
                response_reasonings = response_data.get("reasonings", [])
                
                # Ensure we have enough responses (handle truncation)
                if len(response_values) < len(header_order):
                    print(f"WARNING: Only received {len(response_values)}/{len(header_order)} responses", flush=True)
                    # Pad with None if needed
                    response_values.extend([None] * (len(header_order) - len(response_values)))
                    if len(response_reasonings) < len(header_order):
                        response_reasonings.extend([None] * (len(header_order) - len(response_reasonings)))
                
                # Map responses to header IDs
                for idx, (header_id, header) in enumerate(header_order):
                    if idx < len(response_values) and response_values[idx] is not None:
                        try:
                            response_value = int(response_values[idx])
                            # Validate range
                            if response_value < 0 or response_value > 100:
                                response_value = max(0, min(100, response_value))
                            headers_responses[header_id] = {
                                "spectrum": {"left": header["spectrum"]["left"], "right": header["spectrum"]["right"]},
                                "cue": header["cue"],
                                "response": response_value
                            }
                        except (ValueError, TypeError):
                            headers_responses[header_id] = {
                                "spectrum": {"left": header["spectrum"]["left"], "right": header["spectrum"]["right"]},
                                "cue": header["cue"],
                                "response": None
                            }
                    else:
                        headers_responses[header_id] = {
                            "spectrum": {"left": header["spectrum"]["left"], "right": header["spectrum"]["right"]},
                            "cue": header["cue"],
                            "response": None
                        }
                    
                    # Store reasoning if enabled
                    if self.config["include_reasoning"]:
                        if idx < len(response_reasonings) and response_reasonings[idx]:
                            headers_responses[header_id]["raw_response"] = response_reasonings[idx]
                        else:
                            headers_responses[header_id]["raw_response"] = "No reasoning provided"
                
                game_time = time.time() - game_start_time
                print(f"✓ ({game_time:.2f}s)", flush=True)
        
        except Exception as e:
            game_time = time.time() - game_start_time
            print(f"ERROR: {e} ({game_time:.2f}s)", flush=True)
            # Set all responses to None on error
            for header_id, header in header_order:
                headers_responses[header_id] = {
                    "spectrum": {"left": header["spectrum"]["left"], "right": header["spectrum"]["right"]},
                    "cue": header["cue"],
                    "response": None
                }
                if self.config["include_reasoning"]:
                    headers_responses[header_id]["raw_response"] = f"ERROR: {str(e)}"
        
        game_time = time.time() - game_start_time
        print(f"    Game completed: {game_time:.2f}s total", flush=True)
        
        return headers_responses
    
    def administer_game(self, agent_folder: str, attempt_id: Optional[int] = None, 
                       include_reasoning: Optional[bool] = None) -> Dict[str, Any]:
        """
        Administer Wavelength game to an agent.
        
        Args:
            agent_folder: Path to agent folder (must contain scratch.json)
            attempt_id: Optional attempt ID (auto-incremented if None)
            include_reasoning: Override config for this attempt
        
        Returns:
            Dictionary with game results
        """
        # Override reasoning config if provided
        original_include_reasoning = self.config["include_reasoning"]
        if include_reasoning is not None:
            self.config["include_reasoning"] = include_reasoning
        
        try:
            # Load agent
            agent = GenerativeAgent(agent_folder)
            if not agent.scratch:
                raise ValueError(f"Agent at {agent_folder} has no scratch.json or is empty")
            
            agent_name = agent.get_fullname()
            agent_id = os.path.basename(agent_folder)
            
            game_start_time = time.time()
            print(f"Administering Wavelength game to agent: {agent_name} ({agent_id})", flush=True)
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            
            # Load existing responses
            responses_data = self._load_agent_responses(agent_folder)
            
            # Update agent metadata
            responses_data["agent_metadata"] = {
                "agent_id": agent_id,
                "agent_name": agent_name
            }
            
            # Determine attempt ID
            if attempt_id is None:
                attempt_id = len(responses_data["attempts"]) + 1
            
            # Administer game
            try:
                headers_responses = self._administer_game(agent)
            except Exception as e:
                print(f"    ERROR in game administration: {e}", flush=True)
                raise
            
            # Create attempt record
            attempt = {
                "attempt_id": attempt_id,
                "timestamp": datetime.now().isoformat(),
                "headers": headers_responses,
                "completion_status": "completed"
            }
            
            # Add to attempts list
            responses_data["attempts"].append(attempt)
            
            # Save responses
            self._save_agent_responses(agent_folder, responses_data)
            
            game_time = time.time() - game_start_time
            print(f"\nGame completed in {game_time:.2f}s ({game_time/60:.2f} minutes)", flush=True)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"Saved to: {Path(agent_folder) / 'wavelength_responses.json'}", flush=True)
            
            # Restore original config
            self.config["include_reasoning"] = original_include_reasoning
            
            return {
                "agent_folder": agent_folder,
                "attempt_id": attempt_id,
                "headers_completed": len([h for h in headers_responses.values() if h.get("response") is not None]),
                "total_headers": len(self.game_headers),
                "total_time_seconds": game_time,
                "status": "success"
            }
            
        except Exception as e:
            # Restore original config
            self.config["include_reasoning"] = original_include_reasoning
            raise e
    
    def administer_game_batch(self, agent_folders: List[str], 
                              include_reasoning: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Administer game to multiple agents.
        
        Args:
            agent_folders: List of agent folder paths
            include_reasoning: Override config for this batch
        
        Returns:
            List of result dictionaries
        """
        results = []
        batch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"BATCH WAVELENGTH GAME ADMINISTRATION")
        print(f"{'='*60}")
        print(f"Total agents: {len(agent_folders)}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        for idx, agent_folder in enumerate(agent_folders):
            agent_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"Processing agent {idx + 1}/{len(agent_folders)}: {agent_folder}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            try:
                result = self.administer_game(agent_folder, include_reasoning=include_reasoning)
                result["status"] = "success"
                agent_time = time.time() - agent_start_time
                result["agent_time_seconds"] = agent_time
                results.append(result)
            except Exception as e:
                agent_time = time.time() - agent_start_time
                print(f"ERROR processing {agent_folder}: {e} (took {agent_time:.2f}s)", flush=True)
                results.append({
                    "agent_folder": agent_folder,
                    "status": "error",
                    "error": str(e),
                    "agent_time_seconds": agent_time
                })
        
        batch_time = time.time() - batch_start_time
        successful = [r for r in results if r.get("status") == "success"]
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {batch_time:.2f}s ({batch_time/60:.2f} minutes)")
        if successful:
            avg_time = sum(r.get("agent_time_seconds", 0) for r in successful) / len(successful)
            print(f"Average time per agent: {avg_time:.2f}s ({avg_time/60:.2f} minutes)")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        return results
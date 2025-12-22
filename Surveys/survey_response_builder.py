"""
Survey Response Builder

A modular system to administer surveys to agents and store responses in structured JSON format.
Survey interactions are NOT recorded to memory stream - only responses are saved.

Features:
- Uses existing numerical_resp() method (which uses interview transcript template)
- Supports multiple attempts per agent
- Modular sections (can add/remove sections easily)
- Stores numerical responses + optional reasoning
- Requires all sections to be completed
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation_engine.settings import *
from genagents.genagents import GenerativeAgent


class SurveyResponseBuilder:
    """
    Builds survey responses for agents and stores them in structured JSON format.
    
    Survey interactions are NOT saved to memory stream - only final responses are stored.
    """
    
    def __init__(self, survey_questions_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SurveyResponseBuilder.
        
        Args:
            survey_questions_path: Path to survey_questions.json
            config: Optional configuration dictionary
        """
        self.survey_questions_path = survey_questions_path
        self.survey_data = self._load_survey_questions()
        
        # Default configuration
        default_config = {
            "include_reasoning": True,  # Whether to store raw reasoning text
            "require_all_sections": True,  # Must complete all sections
            "batch_by_section": False,  # If True: send all questions in section at once. If False: one question per inference
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
    
    def _load_survey_questions(self) -> Dict[str, Any]:
        """Load survey questions from JSON file."""
        if not os.path.exists(self.survey_questions_path):
            raise FileNotFoundError(f"Survey questions file not found: {self.survey_questions_path}")
        
        with open(self.survey_questions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_agent_responses(self, agent_folder: str) -> Dict[str, Any]:
        """Load existing survey responses for an agent."""
        responses_path = Path(agent_folder) / "survey_responses.json"
        
        if responses_path.exists():
            # Check if file is empty or invalid
            if os.path.getsize(responses_path) == 0:
                print(f"    Warning: {responses_path} is empty, creating new structure", flush=True)
                return {
                    "survey_metadata": self.survey_data["survey_metadata"],
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
                            "survey_metadata": self.survey_data["survey_metadata"],
                            "agent_metadata": {},
                            "attempts": []
                        }
                    return data
            except json.JSONDecodeError as e:
                print(f"    Warning: {responses_path} contains invalid JSON: {e}", flush=True)
                print(f"    Creating new structure", flush=True)
                return {
                    "survey_metadata": self.survey_data["survey_metadata"],
                    "agent_metadata": {},
                    "attempts": []
                }
        else:
            # Return empty structure
            return {
                "survey_metadata": self.survey_data["survey_metadata"],
                "agent_metadata": {},
                "attempts": []
            }
    
    def _save_agent_responses(self, agent_folder: str, responses_data: Dict[str, Any]):
        """Save survey responses to agent folder."""
        responses_path = Path(agent_folder) / "survey_responses.json"
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(responses_path, 'w', encoding='utf-8') as f:
            json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    def _format_questions_for_llm(self, section: Dict[str, Any]) -> Dict[str, int]:
        """
        Format section questions for LLM numerical_resp() method.
        
        NOTE: This method is kept for backward compatibility but is no longer used
        in _administer_section() which now processes questions one at a time.
        
        Returns:
            Dictionary mapping question text to [min, max] range
        """
        questions_dict = {}
        scale_min = section["scale"]["min"]
        scale_max = section["scale"]["max"]
        
        for question in section["questions"]:
            # Use full question text (already includes prefix if applicable)
            questions_dict[question["question_text"]] = [scale_min, scale_max]
        
        return questions_dict
    
    def _administer_section(self, agent: GenerativeAgent, section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Administer a single section to an agent.
        
        Can operate in two modes:
        - batch_by_section=False: One question per inference (default, more reliable)
        - batch_by_section=True: All questions in section at once (faster, but may truncate)
        
        Returns:
            Dictionary with responses and optional raw_responses
        """
        num_questions = len(section["questions"])
        batch_mode = self.config.get("batch_by_section", False)
        
        if batch_mode:
            return self._administer_section_batch(agent, section)
        else:
            return self._administer_section_single(agent, section)
    
    def _administer_section_single(self, agent: GenerativeAgent, section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Administer a single section to an agent, one question at a time.
        
        Returns:
            Dictionary with responses and optional raw_responses
        """
        num_questions = len(section["questions"])
        print(f"    Administering {section['section_id']} ({num_questions} questions, one at a time)...", flush=True)
        
        responses = {}
        raw_responses = {}
        section_start_time = time.time()
        question_times = []
        
        # Process each question individually
        for i, question_data in enumerate(section["questions"]):
            question_id = question_data["question_id"]
            question_text = question_data["question_text"]
            scale_min = section["scale"]["min"]
            scale_max = section["scale"]["max"]
            
            # Create single-question dictionary
            single_question = {question_text: [scale_min, scale_max]}
            
            # Time the question
            question_start_time = time.time()
            print(f"      Question {i+1}/{num_questions} ({question_id})...", end=" ", flush=True)
            
            try:
                # Get agent response for this single question
                response_data = agent.numerical_resp(single_question, float_resp=False)
                
                # Validate response
                if not response_data or "responses" not in response_data or len(response_data["responses"]) == 0:
                    print(f"ERROR: No response received", flush=True)
                    responses[question_id] = None
                    if self.config["include_reasoning"]:
                        raw_responses[question_id] = "ERROR: No response received"
                    continue
                
                # Extract response (should be single value)
                response_value = response_data["responses"][0]
                responses[question_id] = int(response_value)
                
                # Store reasoning if enabled
                if self.config["include_reasoning"]:
                    if len(response_data.get("reasonings", [])) > 0:
                        raw_responses[question_id] = response_data["reasonings"][0]
                    else:
                        raw_responses[question_id] = "No reasoning provided"
                
                question_time = time.time() - question_start_time
                question_times.append(question_time)
                print(f"✓ ({question_time:.2f}s)", flush=True)
                
            except (ValueError, TypeError, IndexError) as e:
                question_time = time.time() - question_start_time
                question_times.append(question_time)
                print(f"ERROR: {e} ({question_time:.2f}s)", flush=True)
                responses[question_id] = None
                if self.config["include_reasoning"]:
                    raw_responses[question_id] = f"ERROR: {str(e)}"
        
        section_time = time.time() - section_start_time
        avg_time = sum(question_times) / len(question_times) if question_times else 0
        print(f"    {section['section_id']} completed: {section_time:.2f}s total (avg {avg_time:.2f}s per question)", flush=True)
        
        result = {
            "responses": responses
        }
        
        if self.config["include_reasoning"] and raw_responses:
            result["raw_responses"] = raw_responses
        
        return result
    
    def _administer_section_batch(self, agent: GenerativeAgent, section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Administer a single section to an agent, all questions in one inference.
        
        Returns:
            Dictionary with responses and optional raw_responses
        """
        num_questions = len(section["questions"])
        print(f"    Administering {section['section_id']} ({num_questions} questions, all at once)...", flush=True)
        
        responses = {}
        raw_responses = {}
        section_start_time = time.time()
        
        # Format all questions for the section
        scale_min = section["scale"]["min"]
        scale_max = section["scale"]["max"]
        
        # Build questions dictionary - preserve order
        questions_dict = {}
        question_order = []  # Track order to map responses back
        
        for question_data in section["questions"]:
            question_id = question_data["question_id"]
            question_text = question_data["question_text"]
            
            # Include section prefix if available
            if question_data.get("section_prefix"):
                full_question = f"{question_data['section_prefix']} {question_text}"
            else:
                full_question = question_text
            
            questions_dict[full_question] = [scale_min, scale_max]
            question_order.append(question_id)
        
        # Send all questions at once
        print(f"      Sending {num_questions} questions in one inference...", end=" ", flush=True)
        
        try:
            # Get agent response for all questions at once
            response_data = agent.numerical_resp(questions_dict, float_resp=False)
            
            # Validate response
            if not response_data or "responses" not in response_data:
                print(f"ERROR: No response received", flush=True)
                # Set all responses to None
                for question_id in question_order:
                    responses[question_id] = None
                    if self.config["include_reasoning"]:
                        raw_responses[question_id] = "ERROR: No response received"
            else:
                # Map responses back to question_ids
                response_values = response_data["responses"]
                response_reasonings = response_data.get("reasonings", [])
                
                # Ensure we have enough responses (handle truncation)
                if len(response_values) < len(question_order):
                    print(f"WARNING: Only received {len(response_values)}/{len(question_order)} responses", flush=True)
                    # Pad with None if needed
                    response_values.extend([None] * (len(question_order) - len(response_values)))
                    if len(response_reasonings) < len(question_order):
                        response_reasonings.extend([None] * (len(question_order) - len(response_reasonings)))
                
                # Map responses to question IDs
                for idx, question_id in enumerate(question_order):
                    if idx < len(response_values) and response_values[idx] is not None:
                        try:
                            responses[question_id] = int(response_values[idx])
                        except (ValueError, TypeError):
                            responses[question_id] = None
                    else:
                        responses[question_id] = None
                    
                    # Store reasoning if enabled
                    if self.config["include_reasoning"]:
                        if idx < len(response_reasonings) and response_reasonings[idx]:
                            raw_responses[question_id] = response_reasonings[idx]
                        else:
                            raw_responses[question_id] = "No reasoning provided"
                
                section_time = time.time() - section_start_time
                print(f"✓ ({section_time:.2f}s)", flush=True)
        
        except Exception as e:
            section_time = time.time() - section_start_time
            print(f"ERROR: {e} ({section_time:.2f}s)", flush=True)
            # Set all responses to None on error
            for question_id in question_order:
                responses[question_id] = None
                if self.config["include_reasoning"]:
                    raw_responses[question_id] = f"ERROR: {str(e)}"
        
        section_time = time.time() - section_start_time
        print(f"    {section['section_id']} completed: {section_time:.2f}s total", flush=True)
        
        result = {
            "responses": responses
        }
        
        if self.config["include_reasoning"] and raw_responses:
            result["raw_responses"] = raw_responses
        
        return result
    
    def administer_survey(self, agent_folder: str, attempt_id: Optional[int] = None, 
                          sections: Optional[List[str]] = None, 
                          include_reasoning: Optional[bool] = None) -> Dict[str, Any]:
        """
        Administer survey to an agent.
        
        Args:
            agent_folder: Path to agent folder (must contain scratch.json)
            attempt_id: Optional attempt ID (auto-incremented if None)
            sections: Optional list of section IDs to include (None = all sections)
            include_reasoning: Override config for this attempt
        
        Returns:
            Dictionary with survey results
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
            
            survey_start_time = time.time()
            print(f"Administering survey to agent: {agent_name} ({agent_id})", flush=True)
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            
            # Load existing responses
            responses_data = self._load_agent_responses(agent_folder)
            
            # Update agent metadata
            responses_data["agent_metadata"] = {
                "agent_id": agent_id,
                "agent_name": agent_name
            }
            
            # Determine sections to administer
            if sections is None:
                sections_to_administer = [s["section_id"] for s in self.survey_data["sections"]]
            else:
                sections_to_administer = sections
                if self.config["require_all_sections"]:
                    all_section_ids = [s["section_id"] for s in self.survey_data["sections"]]
                    missing = set(all_section_ids) - set(sections_to_administer)
                    if missing:
                        raise ValueError(f"require_all_sections=True but missing sections: {missing}")
            
            # Determine attempt ID
            if attempt_id is None:
                attempt_id = len(responses_data["attempts"]) + 1
            
            # Administer each section
            sections_responses = {}
            for section in self.survey_data["sections"]:
                if section["section_id"] in sections_to_administer:
                    try:
                        section_result = self._administer_section(agent, section)
                        sections_responses[section["section_id"]] = section_result
                    except Exception as e:
                        print(f"    ERROR in section {section['section_id']}: {e}", flush=True)
                        # Re-raise to stop the survey if a section fails
                        raise
            
            # Create attempt record
            attempt = {
                "attempt_id": attempt_id,
                "timestamp": datetime.now().isoformat(),
                "sections": sections_responses,
                "completion_status": "completed"
            }
            
            # Add to attempts list
            responses_data["attempts"].append(attempt)
            
            # Save responses
            self._save_agent_responses(agent_folder, responses_data)
            
            survey_time = time.time() - survey_start_time
            print(f"\nSurvey completed in {survey_time:.2f}s ({survey_time/60:.2f} minutes)", flush=True)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"Saved to: {Path(agent_folder) / 'survey_responses.json'}", flush=True)
            
            # Restore original config
            self.config["include_reasoning"] = original_include_reasoning
            
            survey_time = time.time() - survey_start_time
            
            return {
                "agent_folder": agent_folder,
                "attempt_id": attempt_id,
                "sections_completed": list(sections_responses.keys()),
                "total_questions": sum(len(s["responses"]) for s in sections_responses.values()),
                "total_time_seconds": survey_time,
                "status": "success"
            }
            
        except Exception as e:
            # Restore original config
            self.config["include_reasoning"] = original_include_reasoning
            raise e
    
    def administer_survey_batch(self, agent_folders: List[str], 
                                include_reasoning: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Administer survey to multiple agents.
        
        Args:
            agent_folders: List of agent folder paths
            include_reasoning: Override config for this batch
        
        Returns:
            List of result dictionaries
        """
        results = []
        batch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"BATCH SURVEY ADMINISTRATION")
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
                result = self.administer_survey(agent_folder, include_reasoning=include_reasoning)
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


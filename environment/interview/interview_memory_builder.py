"""
Interview Memory Builder

A modular system to populate agent memory streams through semi-structured interviews,
starting from demographic profiles in scratch.json, following the American Voices Project
interview protocol.

TODO: Future Enhancement - AI Interviewer
The paper mentions an AI interviewer that dynamically generates follow-up questions
tailored to each participant's responses. Currently, we use a fixed interview script.
Future work could implement:
- Dynamic follow-up question generation based on agent responses
- Adaptive interview flow (skip questions, add follow-ups)
- Interviewer reflection/note-taking to guide follow-up questions
See Research Paper/main.md and supplementary materials for details on the AI interviewer implementation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from simulation_engine.settings import *
from genagents.genagents import GenerativeAgent


class InterviewMemoryBuilder:
    """
    Builds agent memory streams through semi-structured interviews.
    
    Features:
    - Modular reflection triggers (periodic, high-importance, manual)
    - Automatic Q&A memory saving
    - Progress tracking and resume capability
    - Batch processing support
    """
    
    def __init__(self, interview_script_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the InterviewMemoryBuilder.
        
        Args:
            interview_script_path: Path to interview_script.json
            config: Optional configuration dictionary. Defaults provided if None.
        """
        self.interview_script_path = interview_script_path
        self.interview_script = self._load_interview_script()
        
        # Default configuration
        default_config = {
            "reflection": {
                "periodic": {"enabled": True, "interval": 5},
                "high_importance": {"enabled": False, "threshold": 80},
                "manual": {"enabled": False, "question_ids": []}
            },
            "interviewer_name": "Interviewer",
            "prompt_template": f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/interview_v1.txt"
        }
        
        # Merge with user config
        if config:
            self._merge_config(default_config, config)
        self.config = default_config
        
        # Reflection tracking
        self.reflection_tracker = {
            "periodic_last": 0,  # Last question ID that triggered periodic reflection
            "high_importance_triggered": set(),  # Question IDs that triggered high-importance reflection
            "manual_triggered": set()  # Question IDs that triggered manual reflection
        }
    
    def _load_interview_script(self) -> List[Dict[str, Any]]:
        """Load interview script from JSON file."""
        with open(self.interview_script_path, 'r') as f:
            return json.load(f)
    
    def _merge_config(self, default: Dict, user: Dict):
        """Recursively merge user config into default config."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def build_memory(self, agent_folder: str, output_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Main workflow: conduct interview and populate agent memory stream.
        
        Args:
            agent_folder: Path to agent folder (must contain scratch.json)
            output_folder: Optional output folder. If None, saves to agent_folder.
            
        Returns:
            Dictionary with interview statistics and results
        """
        if output_folder is None:
            output_folder = agent_folder

        # Ensure memory_stream directory exists for new agents
        memory_stream_path = Path(agent_folder) / "memory_stream"
        memory_stream_path.mkdir(parents=True, exist_ok=True)
        
        # Create or fix empty memory stream files
        embeddings_path = memory_stream_path / "embeddings.json"
        nodes_path = memory_stream_path / "nodes.json"
        
        # Check if files exist and are valid JSON, if not create/repair them
        if not embeddings_path.exists() or embeddings_path.stat().st_size == 0:
            with open(embeddings_path, 'w') as f:
                json.dump({}, f)
        else:
            # Verify it's valid JSON
            try:
                with open(embeddings_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, ValueError):
                # File exists but is invalid JSON, overwrite it
                with open(embeddings_path, 'w') as f:
                    json.dump({}, f)
        
        if not nodes_path.exists() or nodes_path.stat().st_size == 0:
            with open(nodes_path, 'w') as f:
                json.dump([], f)
        else:
            # Verify it's valid JSON
            try:
                with open(nodes_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, ValueError):
                # File exists but is invalid JSON, overwrite it
                with open(nodes_path, 'w') as f:
                    json.dump([], f)
        
        # Load agent
        agent = GenerativeAgent(agent_folder)
        if not agent.scratch:
            raise ValueError(f"Agent at {agent_folder} has no scratch.json or is empty")
        
        # Reset reflection tracker for this agent
        self.reflection_tracker = {
            "periodic_last": 0,
            "high_importance_triggered": set(),
            "manual_triggered": set()
        }
        
        # Note: Progress tracking removed - interview runs to completion without intermediate saves
        start_question_idx = 0  # Always start from beginning
        start_time_step = 1  # Start at 1 (node 0 is empty)
        
        print(f"Starting interview for agent: {agent.get_fullname()}")
        print(f"Starting from question 1 of {len(self.interview_script)}")
        
        # Conduct interview (runs to completion, saves only at the end)
        stats = self._conduct_interview(
            agent, 
            self.interview_script,
            start_time_step,
            start_question_idx
        )
        
        # Final reflection on complete interview
        if len(agent.memory_stream.seq_nodes) > 0:
            print("Triggering final reflection on complete interview...")
            agent.reflect("complete interview and life story", time_step=stats["final_time_step"] + 1, reflection_count=1)
        
        # Final save (only at the end)
        agent.save(output_folder)
        print(f"Interview completed. Agent saved to: {output_folder}")
        
        return {
            "agent_folder": agent_folder,
            "output_folder": output_folder,
            "total_questions": len(self.interview_script),
            "questions_answered": stats["questions_answered"],
            "memories_created": stats["memories_created"],
            "reflections_triggered": stats["reflections_triggered"],
            "final_time_step": stats["final_time_step"]
        }
    
    def _conduct_interview(
        self, 
        agent: GenerativeAgent, 
        interview_script: List[Dict[str, Any]],
        start_time_step: int,
        progress_path: str,
        start_question_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Conduct the interview loop.
        
        Args:
            agent: The generative agent
            interview_script: List of interview questions
            start_time_step: Starting time step (for resume)
            progress_path: Path to progress tracking file
            
        Returns:
            Statistics dictionary
        """
        dialogue_history = []
        time_step = start_time_step
        stats = {
            "questions_answered": 0,
            "memories_created": 0,
            "reflections_triggered": 0,
            "final_time_step": time_step
        }
        
        interviewer_name = self.config["interviewer_name"]
        prompt_template = self.config["prompt_template"]
        
        for question_idx, question_item in enumerate(interview_script):
            question_id = question_item.get("question_id", question_idx + 1)
            question = question_item["question"]
            
            # Add question to dialogue
            dialogue_history.append([interviewer_name, question])
            
            # TODO: Future Enhancement - AI Interviewer with Dynamic Follow-ups
            # Currently using fixed interview script. The paper mentions an AI interviewer
            # that generates personalized follow-up questions based on responses.
            # Future implementation could:
            # - Analyze agent response for interesting points to explore
            # - Generate contextually relevant follow-up questions
            # - Insert follow-ups before moving to next script question
            # - Track interview flow and adapt question sequence
            
            # Get agent response (context is now in the template, not passed as parameter)
            print(f"  Question {question_id}: {question[:60]}...", flush=True)
            agent_response = agent.utterance(
                dialogue_history, 
                context="",  # Empty context since it's in the template
                prompt_template=prompt_template
            )
            
            # Add response to dialogue
            agent_name = agent.get_fullname()
            dialogue_history.append([agent_name, agent_response])
            
            # Save Q&A as memory
            self._save_qa_memory(agent, question, agent_response, time_step)
            stats["memories_created"] += 1
            
            # Get importance score of the memory we just created
            last_node = agent.memory_stream.seq_nodes[-1]
            current_importance = last_node.importance
            
            # Check reflection triggers
            reflection_triggered = self._check_reflection_triggers(
                agent, question_id, current_importance, time_step
            )
            if reflection_triggered:
                stats["reflections_triggered"] += 1
            
            # Increment time step
            time_step += 1
            stats["questions_answered"] += 1
            stats["final_time_step"] = time_step
        
        return stats
    
    def _save_qa_memory(self, agent: GenerativeAgent, question: str, response: str, time_step: int):
        """
        Format and save Q&A pair as observation memory.
        
        Args:
            agent: The generative agent
            question: Interview question
            response: Agent response
            time_step: Current time step
        """
        agent_name = agent.get_fullname()
        interviewer_name = self.config["interviewer_name"]
        qa_content = f"{interviewer_name}: {question}\n\n{agent_name}: {response}\n\n"
        agent.remember(qa_content, time_step=time_step)
    
    def _check_reflection_triggers(
        self, 
        agent: GenerativeAgent, 
        question_id: int, 
        current_importance: int,
        time_step: int
    ) -> bool:
        """
        Check if reflection should be triggered based on configured strategies.
        
        Args:
            agent: The generative agent
            question_id: Current question ID
            current_importance: Importance score of last memory
            time_step: Current time step
            
        Returns:
            True if reflection was triggered, False otherwise
        """
        reflection_config = self.config["reflection"]
        triggered = False
        
        # 1. Periodic reflection
        if reflection_config["periodic"]["enabled"]:
            interval = reflection_config["periodic"]["interval"]
            if question_id - self.reflection_tracker["periodic_last"] >= interval:
                anchor = f"recent interview topics from questions {question_id - interval + 1} to {question_id}"
                self._trigger_reflection(agent, anchor, time_step)
                self.reflection_tracker["periodic_last"] = question_id
                triggered = True
                print(f"    → Periodic reflection triggered (every {interval} questions)", flush=True)
        
        # 2. High-importance reflection
        if reflection_config["high_importance"]["enabled"]:
            threshold = reflection_config["high_importance"]["threshold"]
            if current_importance >= threshold and question_id not in self.reflection_tracker["high_importance_triggered"]:
                # Get the question text for anchor
                question_item = next((q for q in self.interview_script if q.get("question_id") == question_id), None)
                if question_item:
                    question_text = question_item["question"]
                    anchor = f"important topic: {question_text[:100]}"
                    self._trigger_reflection(agent, anchor, time_step)
                    self.reflection_tracker["high_importance_triggered"].add(question_id)
                    triggered = True
                    print(f"    → High-importance reflection triggered (importance: {current_importance})")
        
        # 3. Manual reflection
        if reflection_config["manual"]["enabled"]:
            manual_question_ids = reflection_config["manual"].get("question_ids", [])
            if question_id in manual_question_ids and question_id not in self.reflection_tracker["manual_triggered"]:
                question_item = next((q for q in self.interview_script if q.get("question_id") == question_id), None)
                if question_item:
                    question_text = question_item["question"]
                    anchor = f"key topic: {question_text[:100]}"
                    self._trigger_reflection(agent, anchor, time_step)
                    self.reflection_tracker["manual_triggered"].add(question_id)
                    triggered = True
                    print(f"    → Manual reflection triggered (question {question_id})")
        
        return triggered
    
    def _trigger_reflection(self, agent: GenerativeAgent, anchor: str, time_step: int):
        """
        Execute reflection on the agent.
        
        Args:
            agent: The generative agent
            anchor: Reflection anchor/topic
            time_step: Current time step
        """
        # Use reflection_count=1 to generate a single synthesized reflection
        agent.reflect(anchor, time_step=time_step, reflection_count=1)
    
    def _load_progress(self, progress_path: str) -> Dict[str, Any]:
        """Load progress from file if it exists."""
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self, progress_path: str, question_idx: int, time_step: int):
        """Save progress to file."""
        progress = {
            "last_question_idx": question_idx,
            "last_time_step": time_step
        }
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def build_memory_batch(
        self, 
        agent_folders: List[str], 
        output_base_path: Optional[str] = None,
        parallel: bool = False,
        num_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple agents in batch.
        
        Args:
            agent_folders: List of agent folder paths
            output_base_path: Optional base path for outputs. If None, uses agent folders.
            parallel: Whether to process in parallel (not yet implemented)
            num_workers: Number of parallel workers (if parallel=True)
            
        Returns:
            List of result dictionaries, one per agent
        """
        results = []
        
        for idx, agent_folder in enumerate(agent_folders):
            print(f"\n{'='*60}")
            print(f"Processing agent {idx + 1}/{len(agent_folders)}: {agent_folder}")
            print(f"{'='*60}")
            sys.stdout.flush()  # Ensure real-time output
            
            try:
                output_folder = None
                if output_base_path:
                    agent_id = os.path.basename(agent_folder)
                    output_folder = os.path.join(output_base_path, agent_id)
                
                result = self.build_memory(agent_folder, output_folder)
                result["status"] = "success"
                results.append(result)
                
            except Exception as e:
                print(f"ERROR processing {agent_folder}: {e}")
                results.append({
                    "agent_folder": agent_folder,
                    "status": "error",
                    "error": str(e)
                })
        
        return results


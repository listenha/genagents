#!/usr/bin/env python3
"""
Test script for InterviewMemoryBuilder.
Tests with a single agent to verify the workflow.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environment.interview.interview_memory_builder import InterviewMemoryBuilder


def test_single_agent():
    """Test interview builder with a single agent."""
    
    # Paths
    script_dir = Path(__file__).parent
    interview_script_path = script_dir / "interview_script.json"
    agent_folder = project_root / "agent_bank/populations/gss_agents/0000"
    
    if not interview_script_path.exists():
        print(f"Error: Interview script not found at {interview_script_path}")
        print("Please run extract_interview_script.py first")
        return False
    
    if not (agent_folder / "scratch.json").exists():
        print(f"Error: Agent not found at {agent_folder}")
        return False
    
    # Configuration
    config = {
        "reflection": {
            "periodic": {"enabled": True, "interval": 5},
            "high_importance": {"enabled": True, "threshold": 80},
            "manual": {"enabled": True, "question_ids": [1, 10]}  # Test with first and 10th question
        },
        "save_interval": 5  # Save every 5 questions for testing
    }
    
    # Create builder
    builder = InterviewMemoryBuilder(str(interview_script_path), config)
    
    # Test with first 3 questions only (for quick testing)
    print("=" * 60)
    print("TESTING: Interview Memory Builder")
    print("=" * 60)
    print(f"Agent: {agent_folder}")
    print(f"Interview script: {interview_script_path}")
    print(f"Testing with first 3 questions only")
    print("=" * 60)
    
    # Temporarily limit script for testing
    original_script = builder.interview_script
    builder.interview_script = original_script[:3]
    
    try:
        # Run interview
        result = builder.build_memory(str(agent_folder))
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Status: {result.get('status', 'success')}")
        print(f"Questions answered: {result['questions_answered']}")
        print(f"Memories created: {result['memories_created']}")
        print(f"Reflections triggered: {result['reflections_triggered']}")
        print("=" * 60)
        
        # Verify memory stream
        from genagents.genagents import GenerativeAgent
        agent = GenerativeAgent(str(agent_folder))
        print(f"\nAgent memory stream has {len(agent.memory_stream.seq_nodes)} nodes")
        
        if len(agent.memory_stream.seq_nodes) > 0:
            print("\nFirst memory node:")
            first_node = agent.memory_stream.seq_nodes[0]
            print(f"  Type: {first_node.node_type}")
            print(f"  Importance: {first_node.importance}")
            print(f"  Content preview: {first_node.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original script
        builder.interview_script = original_script


if __name__ == "__main__":
    success = test_single_agent()
    sys.exit(0 if success else 1)


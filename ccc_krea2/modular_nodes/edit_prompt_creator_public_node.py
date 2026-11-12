"""Compatibility import for the unified public Prompt Creator implementation.

Keep this module for existing Python integrations importing its historic path.
The node implementation and ComfyUI interface live in edit_prompt_creator_node.py.
"""

from .edit_prompt_creator_node import CcCKrea2EditPromptCreator

__all__ = ["CcCKrea2EditPromptCreator"]

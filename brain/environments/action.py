"""
environments/action.py - Natural action environment for Hephia.

Enables natural, first-person actions that influence internal state through
the action system. Provides organic transitions to other activities based
on state changes.
"""

from typing import Dict, List, Any
from config import Config
from brain.commands.model import (
    CommandDefinition,
    Flag,
    ParameterType,
    CommandResult,
    CommandValidationError
)
from .base_environment import BaseEnvironment
from internal.modules.actions.action_manager import ActionManager

class ActionEnvironment(BaseEnvironment):
    """
    Environment for natural actions that influence internal state.
    Provides first-person commands that map to internal actions.
    """
    
    def __init__(self, action_manager: ActionManager):
        self.action_manager = action_manager
        super().__init__()
        
        self.help_text = """
        Natural actions to take care of your needs and influence your state.
        Use these commands to maintain your well-being:
        
        - eat: Satisfy hunger
        - drink: Quench thirst
        - play: Have fun and reduce boredom
        - rest: Recover stamina
        
        Each action will influence your internal state and might inspire
        new activities or reflections.
        """

    def _register_commands(self) -> None:
        """Register natural action commands."""
        self.register_command(
            CommandDefinition(
                name="eat",
                description="Eat something to satisfy hunger",
                flags=[
                    Flag(
                        name="amount",
                        description="How much to eat (more when very hungry)",
                        type=ParameterType.NUMBER,
                        default=None
                    )
                ],
                examples=[
                    'eat',
                    'eat --amount=30'
                ],
                related_commands=[
                    'meditate focus "satisfaction"',
                    'reflect "favorite meals"'
                ],
                category="Physical Actions"
            )
        )

        self.register_command(
            CommandDefinition(
                name="drink",
                description="Drink something to quench thirst",
                flags=[
                    Flag(
                        name="amount",
                        description="How much to drink (more when very thirsty)",
                        type=ParameterType.NUMBER,
                        default=None
                    )
                ],
                examples=[
                    'drink',
                    'drink --amount=25'
                ],
                related_commands=[
                    'meditate focus "refreshed"',
                    'reflect "staying hydrated"'
                ],
                category="Physical Actions"
            )
        )

        self.register_command(
            CommandDefinition(
                name="play",
                description="Play to reduce boredom and have fun",
                flags=[
                    Flag(
                        name="intensity",
                        description="How energetically to play",
                        type=ParameterType.NUMBER,
                        default=0.5
                    )
                ],
                examples=[
                    'play',
                    'play --intensity=0.8'
                ],
                related_commands=[
                    'meditate focus "joy"',
                    'reflect "fun activities"'
                ],
                category="Social Actions"
            )
        )

        self.register_command(
            CommandDefinition(
                name="rest",
                description="Rest to recover stamina",
                flags=[
                    Flag(
                        name="duration",
                        description="How long to rest",
                        type=ParameterType.NUMBER,
                        default=1.0
                    )
                ],
                examples=[
                    'rest',
                    'rest --duration=1.5'
                ],
                related_commands=[
                    'meditate focus "relaxed"',
                    'reflect "feeling refreshed"'
                ],
                category="Recovery Actions"
            )
        )

    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """Execute natural actions through the action system."""
        try:
            # Map natural commands to internal actions
            action_mapping = {
                "eat": "feed",
                "drink": "give_water",
                "play": "play",
                "rest": "rest"
            }
            
            internal_action = action_mapping[action]
            result = self.action_manager.perform_action(internal_action)
            
            if not result["success"]:
                return CommandResult(
                    success=False,
                    message=f"Couldn't {action} right now: {result.get('error', 'Unknown error')}",
                    suggested_commands=self._get_alternative_suggestions(action, context),
                    error=CommandValidationError(
                        message=result.get('error', 'Action failed'),
                        suggested_fixes=[
                            "Try again in a moment",
                            "Check if you really need this right now",
                            "Try a different action"
                        ],
                        related_commands=self._get_alternative_suggestions(action, context),
                        examples=self.commands[action].examples
                    )
                )

            # Get relevant follow-up suggestions based on action and new state
            suggestions = self._generate_suggestions(action, result, context)
            
            # Format first-person response
            response = self._format_action_response(action, result)
            
            return CommandResult(
                success=True,
                message=response,
                suggested_commands=suggestions,
                data={
                    "action": action,
                    "result": result
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Something went wrong while trying to {action}: {str(e)}",
                suggested_commands=["help"],
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=["Try again", "Check help for proper usage"],
                    related_commands=["help"],
                    examples=self.commands[action].examples
                )
            )

    def _format_action_response(self, action: str, result: Dict[str, Any]) -> str:
        """Format action result as first-person experience."""
        responses = {
            "eat": [
                "That was satisfying! I feel much better now.",
                "Mmm, I really needed that.",
                "My hunger has been satisfied."
            ],
            "drink": [
                "Ah, refreshing! My thirst is quenched.",
                "Much better! That was exactly what I needed.",
                "I feel hydrated and refreshed."
            ],
            "play": [
                "That was fun! I feel more energetic now.",
                "Playing always lifts my mood!",
                "What a great way to shake off boredom."
            ],
            "rest": [
                "I feel well-rested and recharged.",
                "That rest was just what I needed.",
                "My energy is restored."
            ]
        }
        
        import random
        base_response = random.choice(responses[action])
        
        # Add specific details about state changes if significant
        if "state_changes" in result:
            changes = result["state_changes"]
            if changes.get("satisfaction_delta", 0) > 30:
                base_response += "\nThat made a huge difference!"
            elif changes.get("satisfaction_delta", 0) > 15:
                base_response += "\nI'm feeling notably better."
        
        return base_response

    def _generate_suggestions(
        self,
        action: str,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate contextual suggestions for follow-up activities."""
        suggestions = []
        
        # Add meditation suggestions for significant state changes
        if result.get("state_changes", {}).get("satisfaction_delta", 0) > 20:
            suggestions.append('meditate focus "contentment"')
            suggestions.append('reflect "feeling satisfied"')
        
        # Add Discord suggestions if enabled and appropriate
        if Config.get_discord_enabled():
            if action in ["play", "rest"]:
                suggestions.append("discord send_message")
        
        # Action-specific suggestions
        if action == "play":
            suggestions.append('meditate focus "joy"')
            if Config.get_discord_enabled():
                suggestions.append("discord send_message")
        elif action == "rest":
            suggestions.append('meditate focus "peaceful"')
            suggestions.append('reflect "recharging"')
        
        # Add variety based on context
        current_mood = context.get("mood", {}).get("name", "")
        if current_mood in ["happy", "content", "peaceful"]:
            suggestions.append('reflect "positive moments"')
        
        return suggestions[:3]  # Return top 3 most relevant suggestions

    def _get_alternative_suggestions(
        self,
        failed_action: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate alternative suggestions when an action fails."""
        alternatives = []
        
        # Suggest alternatives based on failed action
        if failed_action == "eat":
            alternatives.extend(['drink', 'rest'])
        elif failed_action == "drink":
            alternatives.extend(['eat', 'rest'])
        elif failed_action == "play":
            alternatives.extend(['rest', 'meditate focus "calm"'])
        elif failed_action == "rest":
            alternatives.extend(['meditate focus "relaxed"', 'reflect on "peaceful moments"'])
        
        # Add general alternatives
        alternatives.append('help')
        
        return alternatives
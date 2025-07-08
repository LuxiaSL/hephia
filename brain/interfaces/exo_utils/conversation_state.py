from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional
from config import Config
from loggers import BrainLogger

class MessageRole(str, Enum):
    """Defines the possible roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

class ConversationPair(BaseModel):
    """Represents an assistant-user message pair."""
    assistant: Message
    user: Message
    
    @field_validator('assistant')
    @classmethod
    def validate_assistant_role(cls, v):
        """Ensure assistant message has the correct role."""
        if v.role != MessageRole.ASSISTANT:
            raise ValueError("First message in pair must have assistant role")
        return v
        
    @field_validator('user')
    @classmethod
    def validate_user_role(cls, v):
        """Ensure user message has the correct role."""
        if v.role != MessageRole.USER:
            raise ValueError("Second message in pair must have user role")
        return v

class ConversationState(BaseModel):
    """
    Manages the state of a conversation with strict structural validation.
    
    A conversation consists of:
    - A system message (instructions/context)
    - An optional initial user message (for the first prompt)
    - A list of assistant-user exchange pairs
    
    This ensures the conversation always follows the pattern:
    [system, assistant, user, assistant, user, ...]
    And always ends with a user message, ready for the next assistant response.
    """
    system_message: Message
    initial_user_message: Optional[Message] = None
    pairs: List[ConversationPair] = Field(default_factory=list)
    
    @field_validator('system_message')
    @classmethod
    def validate_system_role(cls, v):
        """Ensure system message has the correct role."""
        if v.role != MessageRole.SYSTEM:
            raise ValueError("System message must have system role")
        return v
    
    @field_validator('initial_user_message')
    @classmethod
    def validate_initial_user_role(cls, v):
        """Ensure initial user message has the correct role."""
        if v is not None and v.role != MessageRole.USER:
            raise ValueError("Initial user message must have user role")
        return v
    
    @classmethod
    def create_initial(cls, system_content: str, user_content: str) -> "ConversationState":
        """
        Create a new conversation with just system and initial user message.
        
        Args:
            system_content: Content for the system message
            user_content: Content for the initial user message
            
        Returns:
            A new ConversationState with just system and user messages
        """
        return cls(
            system_message=Message(role=MessageRole.SYSTEM, content=system_content),
            initial_user_message=Message(role=MessageRole.USER, content=user_content)
        )
    
    @classmethod
    def from_message_list(cls, messages: List[Dict[str, str]]) -> "ConversationState":
        """
        Convert a traditional message list to a ConversationState.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
        
        Returns:
            A validated ConversationState object
            
        Raises:
            ValueError: If the message list doesn't follow the expected pattern
        """
        if not messages:
            raise ValueError("Cannot create ConversationState from empty message list")
            
        # Extract system message
        if messages[0]['role'] != 'system':
            raise ValueError("First message must be a system message")
            
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=messages[0]['content']
        )
        
        # Process remaining messages as pairs
        pairs = []
        i = 1
        
        # Check for special case: only system + user (initialization state)
        if len(messages) == 2 and messages[1]['role'] == 'user':
            return cls(
                system_message=system_message,
                initial_user_message=Message(
                    role=MessageRole.USER,
                    content=messages[1]['content']
                )
            )
        
        # Process remaining messages as pairs
        initial_user_message = None
        pairs = []
        i = 1
        
        # Handle special case: if we start with a user message after system
        if len(messages) > 1 and messages[1]['role'] == 'user':
            initial_user_message = Message(
                role=MessageRole.USER,
                content=messages[1]['content']
            )
            i = 2
        
        # Process remaining messages as pairs
        while i < len(messages) - 1:
            if messages[i]['role'] != 'assistant' or messages[i+1]['role'] != 'user':
                raise ValueError(f"Expected assistant-user pair at positions {i},{i+1}")
                
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=messages[i]['content']
            )
            
            user_msg = Message(
                role=MessageRole.USER,
                content=messages[i+1]['content']
            )
            
            pairs.append(ConversationPair(assistant=assistant_msg, user=user_msg))
            i += 2
            
        # Check if we have a trailing assistant message
        if i == len(messages) - 1 and messages[i]['role'] == 'assistant':
            raise ValueError("Conversation cannot end with an assistant message")
            
        return cls(
            system_message=system_message,
            initial_user_message=initial_user_message,
            pairs=pairs
        )
    
    def to_message_list(self) -> List[Dict[str, str]]:
        """
        Convert the conversation state to a traditional message list format.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        messages = [{"role": "system", "content": self.system_message.content}]
        
        # Add initial user message if present
        if self.initial_user_message:
            messages.append({
                "role": "user",
                "content": self.initial_user_message.content
            })
        
        # Add all pairs
        for pair in self.pairs:
            messages.append({
                "role": "assistant",
                "content": pair.assistant.content
            })
            messages.append({
                "role": "user",
                "content": pair.user.content
            })
            
        return messages
    
    def add_exchange(self, assistant_content: str, user_content: str, 
                    assistant_metadata: Optional[Dict[str, Any]] = None,
                    user_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a complete assistant-user exchange to the conversation.
        
        Args:
            assistant_content: Content of the assistant's message
            user_content: Content of the user's message
            assistant_metadata: Optional metadata for assistant message
            user_metadata: Optional metadata for user message
        """
        BrainLogger.debug(f"Adding exchange: {assistant_content[:25]} | {user_content[:25]}")
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_content,
            metadata=assistant_metadata or {}
        )
        
        user_msg = Message(
            role=MessageRole.USER,
            content=user_content,
            metadata=user_metadata or {}
        )
        
        self.pairs.append(ConversationPair(
            assistant=assistant_msg,
            user=user_msg
        ))
    
    def get_contextual_history(self, context: str) -> List[Dict[str, str]]:
        """
        Get message history with ephemeral context injected for LLM processing.
        
        Args:
            context: Context information to inject
            
        Returns:
            Message list with context injected after system message
        """
        messages = self.to_message_list()
        
        # Insert context as second message (after system message)
        return [
            messages[0],  # System message
            {"role": "system", "content": context},  # Injected context
            *messages[1:]  # Rest of conversation
        ]
    
    def prune_last_exchange(self) -> bool:
        """
        Remove the most recent exchange (assistant-user pair).
        
        Returns:
            True if successful, False if no pairs to remove
        """
        if not self.pairs:
            return False
            
        self.pairs.pop()
        return True
    
    def trim_to_size(self, max_pairs: int) -> int:
        """
        Limit conversation to maximum number of exchange pairs by removing oldest.
        
        Args:
            max_pairs: Maximum number of pairs to keep
            
        Returns:
            Number of pairs removed
        """
        if len(self.pairs) <= max_pairs:
            return 0
            
        pairs_to_remove = len(self.pairs) - max_pairs
        self.pairs = self.pairs[-max_pairs:]
        return pairs_to_remove
    
    def get_recent_content(self, num_pairs: int = 3) -> str:
        """
        Get concatenated content from recent exchanges for context retrieval.
        
        Args:
            num_pairs: Number of recent pairs to include
            
        Returns:
            Concatenated string of recent exchanges
        """
        recent_pairs = self.pairs[-num_pairs:] if self.pairs else []
        content_parts = []
        
        for pair in recent_pairs:
            content_parts.append(f"{Config.get_cognitive_model()}: {pair.assistant.content}")
            content_parts.append(f"exo: {pair.user.content}")
            
        return "\n".join(content_parts)
    
    def is_valid(self) -> bool:
        """
        Check if the conversation state is structurally valid.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Convert to message list and back as validation
            messages = self.to_message_list()
            test_state = ConversationState.from_message_list(messages)
            return True
        except Exception:
            return False
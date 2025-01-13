# Event Catalog

This document lists all events used in the Hephia project.

(horribly out of date: need to use this idea in the future, but don't trust this file to be whole)

## Needs Module

### need:changed
- **Source**: NeedsManager (modules/needs/needs_manager.py)
- **Data**: 
  - `need_name`: str
  - `old_value`: float
  - `new_value`: float
- **Description**: Dispatched when a need's value changes.

### need:decay_rate_changed
- **Source**: NeedsManager (modules/needs/needs_manager.py)
- **Data**:
  - `need_name`: str
  - `new_base_rate`: float (when base rate is changed)
  - `new_multiplier`: float (when multiplier is changed)
- **Description**: Dispatched when a need's decay rate or multiplier is changed.

## Actions Module

### action:started
- **Source**: ActionManager (modules/actions/action_manager.py)
- **Data**:
  - `action_name`: str
- **Description**: Dispatched when an action is started.

### action:completed
- **Source**: ActionManager (modules/actions/action_manager.py)
- **Data**:
  - `action_name`: str
  - `result`: dict (action-specific result data)
- **Description**: Dispatched when an action is completed.

### action:{action_name}:started
- **Source**: Specific Action classes (e.g., FeedAction, PlayAction)
- **Data**:
  - `action_name`: str
- **Description**: Dispatched when a specific action starts.

### action:{action_name}:completed
- **Source**: Specific Action classes (e.g., FeedAction, PlayAction)
- **Data**:
  - `action_name`: str
  - `result`: dict (action-specific result data)
- **Description**: Dispatched when a specific action is completed.

## Behaviors Module

### behavior:changed
- **Source**: BehaviorManager (modules/behaviors/behavior_manager.py)
- **Data**:
  - `old_behavior`: str
  - `new_behavior`: str
- **Description**: Dispatched when the pet's behavior changes.

### behavior:{behavior_name}:started
- **Source**: Specific Behavior classes (e.g., IdleBehavior, WalkBehavior)
- **Description**: Dispatched when a specific behavior starts.

### behavior:{behavior_name}:updated
- **Source**: Specific Behavior classes (e.g., IdleBehavior, WalkBehavior)
- **Description**: Dispatched when a specific behavior is updated.

### behavior:{behavior_name}:stopped
- **Source**: Specific Behavior classes (e.g., IdleBehavior, WalkBehavior)
- **Description**: Dispatched when a specific behavior stops.

### behavior:{behavior_name}:modifiers_applied
- **Source**: Specific Behavior classes (e.g., IdleBehavior, WalkBehavior)
- **Data**:
  - `base_modifiers`: dict
  - `multiplier_modifiers`: dict
- **Description**: Dispatched when behavior-specific need modifiers are applied.

### behavior:{behavior_name}:modifiers_removed
- **Source**: Specific Behavior classes (e.g., IdleBehavior, WalkBehavior)
- **Data**:
  - `base_modifiers`: dict
  - `multiplier_modifiers`: dict
- **Description**: Dispatched when behavior-specific need modifiers are removed.

## Emotions Module

### emotion:new
- **Source**: EmotionalProcessor (modules/emotions/emotional_processor.py)
- **Data**:
  - `emotion`: Emotion
- **Description**: Dispatched when a significant new emotion is generated.
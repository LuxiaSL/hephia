# Memory System Implementation

Welcome to the **Memory System Implementation**, a sophisticated framework designed to model both **bodily/emotional** (somatic) and **cognitive/semantic** experiences. Inspired by the intricate workings of human memory, this system captures how experiences form, decay, and interconnect over time, enabling intelligent agents to learn, adapt, and interact with their environment in a nuanced and human-like manner.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Key Components](#key-components)
   - [MemorySystem](#memorysystem)
   - [BodyMemory](#bodymemory)
   - [CognitiveMemory](#cognitivememory)
3. [Data Structures](#data-structures)
   - [BodyMemoryNode](#bodymemorynode)
   - [CognitiveMemoryNode](#cognitivememorynode)
4. [Unique Features](#unique-features)
   - [Dual-Module Architecture](#dual-module-architecture)
   - [Advanced Ghosting Mechanics with Revival Capability](#advanced-ghosting-mechanics-with-revival-capability)
   - [Comprehensive Similarity Metrics](#comprehensive-similarity-metrics)
   - [Event-Driven Architecture with Conflict Detection and Synthesis](#event-driven-architecture-with-conflict-detection-and-synthesis)
   - [Echo Mechanism for Memory Reactivation and Influence](#echo-mechanism-for-memory-reactivation-and-influence)
   - [Contextual Modifiers Influencing Memory Formation and Decay](#contextual-modifiers-influencing-memory-formation-and-decay)
   - [Integration with Language Models for Semantic Processing](#integration-with-language-models-for-semantic-processing)
   - [Resilience Through Conflict-Based Synthesis and Merging](#resilience-through-conflict-based-synthesis-and-merging)
   - [Traceable Synthesis and Resurrected Memories](#traceable-synthesis-and-resurrected-memories)
5. [Memory Formation & Updating](#memory-formation--updating)
   - [Body Memory Formation](#body-memory-formation)
   - [Cognitive Memory Formation](#cognitive-memory-formation)
6. [Decay & Ghosting Mechanics](#decay--ghosting-mechanics)
7. [Connections & Similarity](#connections--similarity)
8. [Query & Retrieval](#query--retrieval)
9. [Conflict Detection & Synthesis](#conflict-detection--synthesis)
10. [Event Dispatching](#event-dispatching)
11. [Database & Persistence](#database--persistence)
---

## High-Level Overview

The **Memory System** is engineered to emulate the dual nature of human memory by segregating it into two specialized modules:

- **BodyMemory**: Captures raw, emotional, and physical experiences.
- **CognitiveMemory**: Manages high-level, semantic interpretations and knowledge.

This division allows the system to handle both immediate, visceral experiences and abstract, conceptual information, mirroring how humans process and retain different types of memories. The **MemorySystem** acts as the central coordinator, ensuring seamless integration and interaction between these modules to maintain a coherent and dynamic memory network.

---

## Key Components

### MemorySystem

**_Analogy:_** The **central nervous system** coordinating all memory functions.

**_Description:_**  
The `MemorySystem` class serves as the backbone of the framework, integrating both `BodyMemory` and `CognitiveMemory` modules. It periodically updates these modules to manage memory decay, ghosting, and network reorganization. By coordinating the flow of information and overseeing the lifecycle of memory nodes, the `MemorySystem` ensures that the intelligent agent maintains a balanced and evolving understanding of its experiences.

### BodyMemory

**_Analogy:_** The **emotional heart** capturing visceral experiences.

**_Description:_**  
The `BodyMemory` module focuses on recording and managing the agent’s immediate, sensory-driven experiences, such as emotions, physical sensations, and instinctual needs. Each `BodyMemoryNode` acts like a snapshot of a particular emotional or physical state, storing raw data and processed summaries. This subsystem maintains connections between nodes based on similarity and temporal proximity, enabling the agent to recall and relate past emotional experiences to current situations effectively.

### CognitiveMemory

**_Analogy:_** The **thinking brain** handling thoughts and learned knowledge.

**_Description:_**  
The `CognitiveMemory` module manages high-level reasoning and semantic understanding. It stores `CognitiveMemoryNode` instances that encapsulate detailed textual summaries, conceptual interpretations, and semantic embeddings of experiences. Leveraging advanced language models, this subsystem generates meaningful narratives and insights from raw data, facilitating sophisticated memory retrieval and conflict resolution. The interconnected network of cognitive nodes allows the agent to draw on accumulated knowledge, make informed decisions, and synthesize new ideas, akin to the human brain’s capacity to learn and innovate.

---

## Data Structures

### BodyMemoryNode

**_Description:_**  
A `BodyMemoryNode` represents a single instance of an emotional or physical state experienced by the agent. Key attributes include:

- **timestamp**: When the memory was formed.
- **raw_state**: Unprocessed emotional/physical data.
- **processed_state**: Summarized or partially processed state information.
- **strength**: Indicates how strongly this memory is retained, subject to decay over time.
- **ghosted**: Boolean flag indicating if the node has been merged or is in a latent state.
- **connections**: Dictionary mapping other node IDs to connection weights, representing similarity or interaction strength.
- **ghost_nodes / ghost_states**: Structures for managing merged or prior versions of the node.

### CognitiveMemoryNode

**_Description:_**  
A `CognitiveMemoryNode` encapsulates a specific piece of semantic information or a conceptual understanding derived from experiences. Key attributes include:

- **text_content**: The LLM’s interpretation or summary of the experience.
- **embedding**: A vector used for semantic similarity searches.
- **raw_state**: Complete system state at formation.
- **processed_state**: Processed state data, such as summaries or contextual information.
- **strength**: Represents retention level, subject to decay.
- **ghosted**: Indicates if the node has been merged or is in a latent state.
- **connections**: Links to other cognitive nodes, reflecting semantic and contextual relationships.
- **last_echo_time / echo_dampening**: Attributes managing the echo mechanism for memory reactivation.
- **semantic_context**: Additional semantic metadata.
- **formation_source**: Event or trigger that created this memory.

---

## Unique Features

The Memory System distinguishes itself through several **innovative mechanisms and unique components** that enhance its capability to model human-like memory processes effectively.

### Dual-Module Architecture

**_What It Is:_**  
The system is divided into two specialized modules: `BodyMemory` and `CognitiveMemory`.

**_Why It's Unique:_**  
Most memory systems focus solely on either emotional/sensory data or cognitive/semantic information. By **integrating both**, this system mirrors the dual nature of human memory, allowing for **specialized processing and decay mechanics** tailored to the distinct characteristics of somatic and cognitive memories. This separation enhances the system's ability to model complex human-like memory dynamics, enabling richer and more nuanced interactions.

### Advanced Ghosting Mechanics with Revival Capability

**_What It Is:_**  
Memory nodes transition to a "ghosted" state when their strength decays below certain thresholds. Ghosted nodes retain partial connections and data for potential revival.

**_Why It's Unique:_**  
While many systems implement basic decay or forgetting mechanisms, this system introduces a **multi-tiered ghosting process**:

- **Ghost Threshold**: Nodes become ghosted when their strength falls below a specific level.
- **Final Prune Threshold**: Ghosted nodes are permanently removed if they decay further.
- **Revival Threshold**: Ghosted nodes can be resurrected if they regain sufficient strength, allowing for **dynamic memory restoration** based on evolving contexts.

This nuanced approach ensures that memories are neither abruptly lost nor indefinitely preserved, providing a **balanced and flexible memory lifecycle** akin to human memory retention and forgetting.

### Comprehensive Similarity Metrics

**_What It Is:_**  
The system employs **multi-faceted similarity calculations** assessing connections based on semantic similarity, emotional resonance, state alignment, temporal proximity, and strength factors.

**_Why It's Unique:_**  
Most memory systems rely on **single-dimensional similarity measures**. This system's **holistic approach** ensures connections are formed and weighted based on a **comprehensive understanding of similarity**, integrating emotional, cognitive, and contextual factors. This results in a **more nuanced and human-like network of memories**, where multiple aspects of experiences interplay to shape relationships between memories.

### Event-Driven Architecture with Conflict Detection and Synthesis

**_What It Is:_**  
An **event dispatcher** handles memory formation, updates, and interactions, including conflict detection between memory nodes.

**_Why It's Unique:_**  
The **dynamic event-driven approach** allows for **real-time responsiveness** to changes in the agent's state. The **conflict detection and synthesis mechanism** actively monitors for contradictions between strongly connected memories and resolves them by **merging conflicting memories into unified synthesis nodes**. This ensures the memory network remains **consistent and logically coherent**, preventing the accumulation of conflicting or fragmented memories and enhancing the system's ability to adapt and learn from complex experiences.

### Echo Mechanism for Memory Reactivation and Influence

**_What It Is:_**  
Upon retrieval, memories can trigger "echoes" that **reactivate and influence** the agent based on the memory's intensity and relevance.

**_Why It's Unique:_**  
The **echo mechanism** introduces a **feedback loop** where memories can **actively shape current states and behaviors**. Unlike passive retrieval systems, echoes can **dynamically adjust memory strength**, **trigger further memory formations**, and **influence decision-making processes**, emulating how recalling a memory can affect one’s current emotions and actions.

### Contextual Modifiers Influencing Memory Formation and Decay

**_What It Is:_**  
Memory formation and decay rates are influenced by **contextual factors** such as current mood, behavioral state, and cognitive load.

**_Why It's Unique:_**  
The **context-aware decay mechanism** allows the system to **reflect the influence of the agent's current state** on memory retention. This ensures that the memory system **aligns with the agent's experiential context**, offering a more **realistic and adaptive memory lifecycle**.

### Integration with Language Models for Semantic Processing

**_What It Is:_**  
Utilizes **SentenceTransformer** models to generate **semantic embeddings** for textual content, enabling advanced similarity searches and semantic density calculations.

**_Why It's Unique:_**  
By **leveraging state-of-the-art language models**, the Memory System achieves a **high level of semantic understanding** that surpasses traditional keyword-based approaches. This integration facilitates **deep semantic matching**, **enhanced conflict detection**, and **rich memory representations**, making it a powerful tool for intelligent agents requiring sophisticated memory capabilities.

### Resilience Through Conflict-Based Synthesis and Merging

**_What It Is:_**  
When conflicting memories are detected, the system **synthesizes a new memory node** that amalgamates the conflicting information, adjusting the strengths of the original nodes and preserving relational data.

**_Why It's Unique:_**  
This **conflict-based synthesis** approach ensures the Memory System can **resolve inconsistencies** and **evolve its memory network** intelligently. By adjusting the influence of original memories and maintaining the history and connections of merged memories, the system ensures **memory network stability and reliability**.

### Traceable Synthesis and Resurrected Memories

**_What It Is:_**  
Maintains detailed records of how memories have been **synthesized or resurrected**, including their **origins and contributing nodes**.

**_Why It's Unique:_**  
This **traceability** allows for **knowledge continuity**, **enhanced learning**, and **auditability**, ensuring that the memory network remains coherent and interconnected, even as memories undergo complex transformations.

---

## Memory Formation & Updating

### Body Memory Formation

**_Description:_**  
Triggered by events such as `emotion:finished`, `behavior:changed`, `mood:changed`, or `need:changed`, the `BodyMemory` module creates new `BodyMemoryNode` instances when certain thresholds are met. These nodes capture the current raw and processed states, establish initial connections based on similarity with recent nodes, and are persisted in the database.

### Cognitive Memory Formation

**_Description:_**  
Involves a two-step process:
1. **Request** content generation from an external LLM or summarizer.
2. **Complete** memory formation once text content is generated, creating a new `CognitiveMemoryNode` with the LLM’s text summary, embedding vector, and the system’s state snapshot. This node is linked to relevant `BodyMemoryNode` instances or created independently if no suitable context is found.

---

## Decay & Ghosting Mechanics

Both `BodyMemoryNode` and `CognitiveMemoryNode` instances have a `strength` attribute that **decays over time**. When `strength` falls below:

1. **Ghost Threshold**: The node is marked as `ghosted`, indicating it is no longer active but retained for potential revival.
2. **Final Prune Threshold**: If a ghosted node continues to decay, it may be permanently removed or merged into a parent node.

This multi-tiered approach ensures that memories are managed dynamically, maintaining relevance while preventing unnecessary clutter.

---

## Connections & Similarity

Memory nodes maintain **connections** with other nodes based on **similarity scores**, which are calculated using multi-dimensional metrics that assess semantic similarity, emotional resonance, state alignment, temporal proximity, and strength factors. These connections form a **networked memory structure** that allows for efficient retrieval and relationship mapping, enhancing the agent's ability to recall and relate experiences effectively.

---

## Query & Retrieval

The Memory System offers robust **querying and retrieval functionalities**, enabling the intelligent agent to access relevant memories based on criteria such as similarity, recency, or specific semantic content. CognitiveMemory leverages semantic embeddings for advanced similarity searches, allowing for precise and meaningful memory retrieval. Retrieved memories can trigger **echoes**, influencing the agent's current states and behaviors.

---

## Conflict Detection & Synthesis

The system actively monitors for **conflicts** between memory nodes—situations where two memories strongly overlap but contain contradictory information. Upon detecting such conflicts, it triggers **synthesis processes** that merge conflicting memories into unified synthesis nodes. This ensures that the memory network remains **consistent and logically coherent**, enhancing the agent's ability to adapt and learn from complex experiences.

---

## Event Dispatching

An **event-driven architecture** underpins the Memory System, with a central **event dispatcher** managing the flow of events such as memory formation, updates, conflict detection, and synthesis. This architecture allows for **real-time responsiveness** and seamless integration between the `BodyMemory` and `CognitiveMemory` modules, ensuring that memory operations are handled efficiently and coherently.

---

## Database & Persistence

All memory nodes and their relationships are **persistently stored in a SQLite database** (`memory.db` by default). The database schema includes separate tables for `body_memory_nodes`, `cognitive_memory_nodes`, `memory_links`, and `synthesis_relations`. This robust persistence layer ensures data integrity, scalability, and traceability, allowing the Memory System to maintain a comprehensive and enduring memory repository.


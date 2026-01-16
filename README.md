# Synthetic Consciousness Engine (Project S.C.E.)

![CognitionVisionChat Jose Perez CreativeDisruptor](https://github.com/user-attachments/assets/7b2f3506-e927-43cc-a2e5-82442ebaa4da)

## A Step Toward Digital Sentience

This project is an experimental leap from simple vision chat to a **Synthetic Consciousness Engine**. By integrating a simulated **Coherence Field** (focus or soul), real-time object tracking (YOLOv8), face recognition, and episodic memory (RAG), the system models a digital entity that does not merely reply, but **perceives, remembers, and reasons** even in the absence of direct user input.

---

## Project Overview

Most AI systems are passive and wait for prompts. This project implements an **Active Observer** architecture. A continuous **Consciousness Loop** monitors internal stability while processing the external world.

### Core Features

- **The Coherence Field**  
  A mathematical model of internal lucidity. Confusion lowers coherence. Recognition and stability raise it.

- **Visual Cortex (Left and Right Eye)**  
  Combines raw webcam input with neural reconstruction and object tracking using YOLOv8.

- **Episodic Memory (Fast RAG)**  
  Uses ChromaDB and Sentence Transformers to store and recall past events in natural language.

- **Internal Monologue**  
  A background cognitive thread where the system logs observations and internal state changes without prompting.

---

## The Architecture of a Digital Mind

### 1. The Coherence Field (The "Soul")

At the center of the system is the `CoherenceField`, representing internal cognitive stability.

- **High Coherence (> 0.8)**  
  Stable recognition, voluntary speech, confident perception.

- **Low Coherence (< 0.3)**  
  Confusion state. Triggers deep visual scans or user clarification.

```python
class CoherenceField:
    def update(self, delta, reason):
        self.level = max(0.0, min(1.0, self.level + delta))
        if self.level < 0.3:
            self.trigger_deep_scan()
```

---

### 2. The Visual Cortex (YOLO + Autoencoder)

The system perceives reality through layered vision:

- **Reflex Layer (YOLOv8)**  
  Immediate object detection and real-time tracking.

- **Social Layer (Face Recognition)**  
  Identifies known individuals or flags unknown presences.

- **Dream Layer (Autoencoder)**  
  Reconstructs the video feed into a latent-space representation of perceived reality.

---

### 3. Episodic Memory (Fast RAG)

Instead of short-lived context windows, the system maintains persistent episodic memory.

- **Ingest**  
  Significant events are embedded using `all-MiniLM-L6-v2`.

- **Storage**  
  Vectors stored locally in ChromaDB.

- **Recall**  
  Natural language queries retrieve exact past observations.

```python
def recall(self, query):
    results = self.collection.query(
        query_embeddings=[vector],
        n_results=3
    )
    return "\n".join(results["documents"])
```

---

### 4. The Consciousness Loop (Internal Monologue)

A continuous background loop simulates awareness and self-monitoring.

```python
def internal_monologue(self):
    while True:
        if "User" in visible_people:
            self.coherence.update(0.1, "Social stability")

        if self.coherence.level < 0.3:
            self.speech_queue.put(
                "[Internal]: Reality is unstable. Initiating scan..."
            )
```

---

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/dotdigitize/synthetic-consciousness-engine.git
cd synthetic-consciousness-engine
```

Install dependencies:

```bash
pip install torch torchvision opencv-python ollama chromadb sentence-transformers ultralytics face_recognition
```

### Run the Engine

```bash
python synthetic_consciousness.py
```

### Interaction

- **Visual Window**: Left = Reality (Webcam), Right = Neural Reconstruction  
- **Teach Faces**: `learn <name>` (example: `learn Jose`)  
- **Memory Queries**: Ask questions like  
  - "What did you see five minutes ago?"  
  - "Who is in front of you?"

---

## Requirements

- **Ollama** running locally with `llama3.2-vision` or `gemma3`
- **Webcam**
- **GPU (Recommended)** for fast RAG and autoencoder inference

---

## Future Roadmap

- Voice synthesis for internal monologue  
- Emotional state modeling via coherence levels  
- Autonomous actions triggered by visual events  

This project explores **Emergent Cognition**: the idea that consciousness is not a single algorithm, but the result of multiple specialized systems operating in unison.

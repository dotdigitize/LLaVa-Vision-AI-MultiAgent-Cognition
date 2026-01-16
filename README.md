# AI Cognition Vision Chat: Multi-Agent Vision Testbed

![CognitionVisionChat Jose Perez CreativeDisruptor](https://github.com/user-attachments/assets/7b2f3506-e927-43cc-a2e5-82442ebaa4da)

## A Step Toward Real-Time AI Cognition

**Based on the theoretical framework at [CoherenceFieldEquation.org](https://coherencefieldequation.org/)**

This project is an experimental exploration into AI cognition, leveraging convolutional autoencoders, real-time webcam vision, and multi-agent conversational AI models. By merging real-time vision with advanced AI reasoning, the goal is to simulate how artificial agents can perceive, interpret, and respond to the world around them.

This repository serves as the **Vision Test Component** for a larger multi-agent architecture described by the Coherence Field Equation.

---

## Project Overview

Artificial intelligence continues to advance rapidly, yet true AI cognition—where an agent perceives its environment, makes decisions, and communicates intelligently—remains an open frontier. This project explores that frontier by integrating multiple AI subsystems into a unified testbed that:

- Processes real-time visual information from a webcam  
- Uses **Convolutional Autoencoders** to compress and reconstruct visual memory  
- Employs **Universal Vision Processing** (LLaVa, Gemma 3, Llama 3.2 Vision) for semantic scene understanding  
- Generates context-aware conversations using **LLaMA 3.1** models  
- Maintains temporal coherence using **Vector Memory RAG** (ChromaDB)  
- Tracks visual and cognitive stability using the **Coherence Field Equation**  

The result is a proof-of-concept system demonstrating how multiple AI agents can collaborate to produce emergent, visually grounded cognition.

---

## Inspiration and Purpose

### The Vision Behind AI Cognition

Human cognition integrates perception, memory, and communication continuously. This project attempts to simulate a minimal analogue of that process in artificial systems by binding vision, memory, and language into a single loop.

- **Convolutional Autoencoder**  
  Compresses and reconstructs visual input, forming a simplified neural memory.

- **Flexible Vision Processing (LLaVa / Gemma / Llama 3.2)**  
  Images are interpreted using local Vision Language Models. The system is fully model-agnostic and supports any Ollama-compatible VLM.

- **LLaMA 3.1 Agents**  
  Conversational agents ground language responses in current perception and recalled visual memory.

- **Coherence Field Integration**  
  The system acts as an observer, measuring environmental stability through a continuously updated coherence score.

---

## Project Structure & Modes

This repository contains two operational modes.

### 1. Basic Mode (`main.py`)

**The Light Version**  
Used for testing webcam input, Ollama connectivity, and basic vision-chat functionality.

- Autoencoder-based visual memory  
- Vision chat agent  
- Minimal overhead  

Run:
```bash
python main.py
```

### 2. Advanced Mode (`synthetic_consciousness.py`)

**The Full Engine**  
Implements the Synthetic Consciousness architecture with memory, stability tracking, and social awareness.

Features include:
- Face recognition with one-shot learning (`learn <name>`)  
- YOLO-based object tracking for object permanence  
- Vector RAG memory using ChromaDB  
- Coherence Field stability scoring  
- Internal monologue for continuous self-monitoring  

Run:
```bash
python synthetic_consciousness.py
```

---

## How the Code Works

### 1. Convolutional Autoencoder for Memory

```python
class FastConvAutoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
```

### 2. Universal Vision Processing (Model-Agnostic)

```python
VISION_MODEL = "gemma3:27b-it-qat"

def interpret_image_with_vision_model(image_path):
    res = ollama.chat(model=VISION_MODEL, ...)
    return res["message"]["content"]
```

### 3. Visual Memory RAG (Advanced Mode)

The system uses vector embeddings stored in ChromaDB to recall past visual events, enabling temporal continuity and coherence.

### 4. Synthetic Consciousness Loop

- **High Coherence (> 0.8):** Stable recognition and confident perception  
- **Low Coherence (< 0.3):** Deep scans triggered to regain stability  

---

## Getting Started

### Prerequisites

- **Ollama** running locally (`ollama serve`)  
- Required models:
```bash
ollama pull llama3.1:8b
ollama pull gemma3:27b-it-qat   # or llava-phi3
ollama pull nomic-embed-text
```

### Installation

```bash
git clone https://github.com/dotdigitize/ai-cognition-vision-chat.git
cd ai-cognition-vision-chat
pip install -r requirements.txt
```

### Run

```bash
python synthetic_consciousness.py
```

---

## Use Cases

- AI surveillance and environmental stability monitoring  
- Interactive, perception-aware AI companions  
- Research into emergent cognition  
- Experimental validation of the Coherence Field Equation  

---

## Future Work

- Multi-modal input (audio + vision)  
- Long-term persistent memory  
- Autonomous decision-making agents  

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request to discuss ideas or improvements.

For theoretical background, visit **https://coherencefieldequation.org/**

# AI Cognition Vision Chat: Multi-Agent Vision Testbed

![CognitionVisionChat Jose Perez CreativeDisruptor](https://github.com/user-attachments/assets/7b2f3506-e927-43cc-a2e5-82442ebaa4da)

## A Step Toward Real-Time AI Cognition

**Based on the theoretical framework at [CoherenceFieldEquation.org](https://coherencefieldequation.org/)**

This project is an experimental exploration into AI cognition, leveraging convolutional autoencoders, real-time webcam vision, and multi-agent conversational AI models. By merging real-time vision with advanced AI reasoning, the system simulates how artificial agents can perceive, interpret, and respond to the world around them.

This repository serves as the **Vision Test Component** for a larger multi-agent architecture described by the Coherence Field Equation.

---

## Project Overview

This system explores how artificial agents can move beyond passive response and toward active perception by integrating:

- Real-time visual input from a webcam
- Convolutional autoencoders for visual memory
- **Model-agnostic vision processing** (LLaVa, Gemma 3, Llama 3.2 Vision, or any Ollama-compatible VLM)
- Context-aware conversational agents using LLaMA 3.1
- Temporal memory using Vector RAG (ChromaDB)
- Visual stability tracking via the Coherence Field Equation

The result is a proof-of-concept cognitive system that responds based on a persistent and evolving understanding of its environment.

---

## Inspiration and Purpose

### The Vision Behind AI Cognition

Human cognition integrates perception, memory, and language continuously. This project attempts to model a minimal analog of that process in artificial systems.

- **Convolutional Autoencoder**  
  Compresses and reconstructs visual input, forming a simplified neural memory of the environment.

- **Flexible Vision Processing (LLaVa / Gemma / Llama 3.2)**  
  Images are interpreted using local Vision Language Models. The system is fully model-agnostic.

- **LLaMA 3.1 Conversational Agents**  
  Language responses are grounded in current vision and recalled memory.

- **Coherence Field Integration**  
  The system monitors environmental and internal stability, acting as an observer within the Coherence Field framework.

---

## How the Code Works

### 1. Convolutional Autoencoder for Visual Memory

```python
class FastConvAutoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
```

### 2. Real-Time Webcam Processing

```python
def webcam_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        input_tensor = preprocess_frame_for_nn(frame)
        train_autoencoder(input_tensor)
        save_high_res_image(frame, webcam_image_path)
        interpret_image_with_vision_model(webcam_image_path)
        cv2.imshow("Webcam Feed", frame)
```

### 3. Universal Vision Processing (Model-Agnostic)

```python
VISION_MODEL = "gemma3:27b-it-qat"  # or llava-phi3, llama3.2-vision

def interpret_image_with_vision_model(image_path):
    res = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": "Describe the scene:", "images": [image_path]}]
    )
    return res["message"]["content"]
```

### 4. Synthetic Consciousness Features

- Face recognition with one-shot learning
- YOLO-based object tracking
- Vector memory RAG using ChromaDB
- Internal monologue for self-monitoring

```python
class CoherenceField:
    def update(self, delta, reason):
        self.level = max(0.0, min(1.0, self.level + delta))
        if self.level < 0.3:
            print("[Internal]: Reality is unstable. Initiating deep scan...")
```

### 5. Multi-Agent Conversational System

```python
class Agent:
    def respond(self, message, location_description):
        prompt = f"I see: {location_description}\nUser: {message}"
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
```

---

## Use Cases

- Cognitive vision research
- Interactive AI companions
- Environmental stability monitoring
- Coherence Field theory testing

---

## Getting Started

### Installation

```bash
git clone [https://github.com/dotdigitize/ai-cognition-vision-chat.git](https://github.com/dotdigitize/ai-cognition-vision-chat.git)
cd ai-cognition-vision-chat
pip install torch torchvision opencv-python ollama chromadb sentence-transformers ultralytics face_recognition
```

### Run

```bash
python synthetic_consciousness.py
```

---

## Project Structure

```bash
.
├── main.py
├── synthetic_consciousness.py
├── requirements.txt
└── README.md
```

---

## Future Work

- Audio perception and speech synthesis
- Long-term memory persistence
- Autonomous decision-making agents

---

For theoretical background, visit **https://coherencefieldequation.org/**


# AI Cognition Vision Chat (Using Local Langauge Models)
![CognitionVisionChat Jose Perez CreativeDisruptor](https://github.com/user-attachments/assets/7b2f3506-e927-43cc-a2e5-82442ebaa4da)

## A Step Toward Real-Time AI Cognition

This project is an experimental exploration into AI cognition, leveraging the power of convolutional autoencoders, real-time webcam vision, and multi-agent conversational AI models. By merging real-time vision with advanced AI reasoning, we aim to simulate how artificial agents can perceive, interpret, and respond to the world around them.

## Project Overview

Artificial intelligence is becoming increasingly capable of mimicking human-like thought processes, but true AI cognition—where an agent perceives its environment, makes decisions, and communicates intelligently—remains a challenging frontier. This project was conceived to explore these possibilities by integrating various AI components into a cohesive system that:

- Processes real-time visual information from a webcam
- Uses convolutional autoencoders to store and "remember" background images
- Employs LLaVa (Large Language Vision Assistance) to interpret its visual environment
- Generates context-aware conversations using LLaMa 3.1 models, based on what the AI "sees"

The result is a proof of concept for how multiple AI agents can collaborate to simulate a form of emergent cognition, where the AI system doesn’t just respond to inputs but does so based on a dynamic and visual understanding of its environment.

## Inspiration and Purpose

### The Vision Behind AI Cognition

The goal of this project is to simulate a small piece of what human cognition might look like in AI. Humans don’t just passively observe their surroundings—they analyze, process, and respond to the world based on memory, perception, and prior knowledge. Similarly, this system attempts to simulate a rudimentary form of cognitive processing by tying together vision, memory, and communication.

- **Convolutional Autoencoder**: The human brain often simplifies visual input to identify important features. The autoencoder in this project does something similar: it compresses and reconstructs the image, creating a simplified memory that the AI can process.
- **LLaVa Vision Processing**: Vision is a key component of understanding. By running images through the LLaVa system, this AI attempts to "see" and describe its environment in a coherent and detailed manner.
- **LLaMa 3.1 Agents**: Just like how humans use their senses to inform their conversations, the LLaMa-based AI agent uses its visual input to generate more relevant, insightful responses. The multi-agent system enables conversational context by "remembering" the visual environment and generating responses accordingly.

## How the Code Works

### 1. Convolutional Autoencoder for Memory


At the core of this project is a `FastConvolutionalAutoencoder`, a type of neural network designed to reduce the dimensionality of input images (captured by the webcam) and then reconstruct them. This autoencoder acts as the AI’s memory, compressing incoming visual data and allowing the AI to "recall" its simplified form later.

- **Encoder**: Compresses the image into a smaller, more memory-efficient representation.
- **Decoder**: Reconstructs the image back to its original form, or as close as possible, simulating the memory recall process.

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
![convolutional memory sample](https://github.com/user-attachments/assets/37f6a2a3-183d-4f20-b1e9-7d6d973dbce3)

### 2. Real-Time Webcam Feed

A webcam feed continuously streams visual data into the system. This visual input is processed and saved as a high-resolution image for interpretation by the LLaVa system. The frame is also passed through the autoencoder for memory compression.

```python
def webcam_loop():
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        input_tensor = preprocess_frame_for_nn(frame)
        processed_image = train_autoencoder(input_tensor)
        save_high_res_image(frame, webcam_image_path)
        interpret_image_with_llava(webcam_image_path)
        cv2.imshow('Webcam Feed', frame)
```

### 3. LLaVa Vision Processing

LLaVa is used to interpret the webcam feed. The saved high-res image is passed to the LLaVa model, which analyzes it and generates a description of what it "sees." This description is critical for the multi-agent AI’s ability to interact with and respond to its environment.

```python
def interpret_image_with_llava(image_path):
    res = ollama.chat(
        model="llava-phi3",
        messages=[{
            'role': 'user',
            'content': 'Describe the scene in detail:',
            'images': [image_path]
        }]
    )
    llava_observation = res['message']['content']
```

### 4. Multi-Agent System Using LLaMa 3.1

The heart of the interaction lies in the `Agent` class, which is based on the LLaMa 3.1 model. The agent receives visual information from LLaVa, formats it into a natural language prompt, and uses the LLaMa model to generate responses. The agent’s responses are conversational and context-aware, making the interaction feel more lifelike.

```python
class Agent:
    def respond(self, message, location_description):
        prompt = self.create_prompt(location_description) + self.format_conversation_history() + f"User: {message}
{self.name}:"
        prompt += f" I see: {llava_observation}"
        response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
```

## Use Cases

This project serves as a basis for exploring how AI can simulate cognition by processing real-time visual information. Some potential applications include:

- **AI Surveillance**: Real-time AI that observes and comments on an environment, offering insights or monitoring for abnormalities.
- **Interactive AI Companions**: AI systems that can understand their surroundings and engage users in more intelligent, meaningful conversations.
- **Research in AI Cognition**: This system provides a framework to explore how AI can develop cognition by processing and responding to environmental inputs.

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/dotdigitize/ai-cognition-vision-chat.git
cd ai-cognition-vision-chat
```

Install the required libraries:

Use the provided `requirements.txt` to install the necessary dependencies.

```bash
pip install -r requirements.txt
```

### Run the Project

Start the webcam feed and agent chat:

```bash
python main.py
```

### Interact:

- The AI will begin processing the webcam feed and interpreting the scene.
- Engage in conversation with the AI, and it will respond with context-aware observations of its environment.

## Project Structure

```bash
.
├── main.py            # Main Python script for the autoencoder and chat system
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Future Work

This project lays the groundwork for future explorations into AI cognition. Planned enhancements include:

- **Multi-modal integration**: Combining audio input (via microphones) alongside vision to create a more immersive cognitive experience.
- **Advanced Memory Networks**: Implementing long-term memory to allow the AI to remember past interactions and environments across sessions.
- **Autonomous Decision Making**: Moving beyond passive observation, enabling the AI to make decisions based on its understanding of the environment.

## Contributing

Contributions are welcome! If you have ideas or improvements, feel free to submit a pull request or open an issue to discuss your thoughts.

---

This project is a stepping stone into the world of AI cognition and multi-agent systems. By integrating vision, memory, and conversational agents, we hope to simulate how AI can perceive and respond to the world in ways that mimic human-like awareness.

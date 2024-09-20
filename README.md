# AI Cognition Vision Chat

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

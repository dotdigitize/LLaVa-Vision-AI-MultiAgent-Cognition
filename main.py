import torch
import torch.nn as nn
import torch.optim as optim
import cv2  # For webcam access
import time
from PIL import Image
import torchvision.transforms as transforms
import ollama
import threading

# This class defines the architecture of a simple convolutional autoencoder.
# The autoencoder is designed to compress images into a smaller representation (encoding)
# and then reconstruct them back into their original form (decoding).
class FastConvAutoencoder(nn.Module):
    def __init__(self):
        # The init method defines the encoder and decoder layers of the autoencoder.
        # The encoder downsamples the image, and the decoder upscales it back to its original size.
        super(FastConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),  # Convolution layer 1
            nn.ReLU(),  # ReLU activation to introduce non-linearity
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Convolution layer 2
            nn.ReLU()  # ReLU activation
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Transposed Convolution to upsample
            nn.ReLU(),  # ReLU activation
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Final layer to return the image to its original depth (3 channels)
            nn.Sigmoid()  # Sigmoid activation to get pixel values between 0 and 1
        )

    # The forward method defines the forward pass of the network, where the input image is first encoded and then decoded.
    def forward(self, x):
        encoded = self.encoder(x)  # Pass through encoder layers
        decoded = self.decoder(encoded)  # Pass through decoder layers
        return decoded

# This function preprocesses a frame from the webcam into a format suitable for the neural network.
# The frame is converted to a tensor, resized to a fixed size, and then transformed into a batch of 1 image.
def preprocess_frame_for_nn(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize((360, 640)),  # Resize the image to 360x640
        transforms.ToTensor()  # Convert the image to a tensor
    ])
    image_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    return image_tensor

# This function saves a high-resolution frame from the webcam feed to a file.
# It resizes the frame to 1280x720 pixels and converts the BGR format (used by OpenCV) to RGB (used by PIL).
def save_high_res_image(frame, file_path):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and then to PIL
    image_pil = image_pil.resize((1280, 720))  # Resize the image to 1280x720 pixels
    image_pil.save(file_path)  # Save the image

# Global variables used to store observations and control the synchronization of threads.
llava_observation = ""  # Stores the textual description of what the LLaVa model "sees"
llava_observation_lock = threading.Lock()  # Lock to ensure that the observation is updated correctly in a multi-threaded environment
llava_has_seen = threading.Event()  # Event to indicate when LLaVa has processed the first image
stop_event = threading.Event()  # Event to signal the stopping of the webcam loop

# This function sends the saved high-resolution image to the LLaVa model for interpretation.
# It runs in a separate thread to prevent blocking the main execution.
def interpret_image_with_llava(image_path):
    def interpret():
        global llava_observation
        with llava_observation_lock:
            res = ollama.chat(
                model="llava-phi3",  # Using LLaVa model for vision processing
                messages=[{
                    'role': 'user',
                    'content': 'Describe the scene in detail:',  # Prompt sent to the model
                    'images': [image_path]  # Send the image for interpretation
                }]
            )
            llava_observation = res['message']['content']  # Store the model's description of the image
            llava_has_seen.set()  # Signal that the LLaVa model has seen the environment
    threading.Thread(target=interpret).start()  # Start the interpretation in a new thread

# Set up the autoencoder model and optimizer.
# The Adam optimizer is used to adjust the weights of the model based on the loss.
# Mean Squared Error (MSE) loss is used to measure the difference between the original and reconstructed images.
model = FastConvAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
criterion = nn.MSELoss()  # Mean Squared Error loss

# This function trains the autoencoder on a target image for a specified number of iterations.
# After each iteration, the loss between the original and reconstructed images is calculated and the weights are updated.
def train_autoencoder(target_image, iterations=100):
    for iteration in range(iterations):
        optimizer.zero_grad()  # Reset gradients before backpropagation
        reconstructed_image = model(target_image)  # Reconstruct the image using the autoencoder
        loss = criterion(reconstructed_image, target_image)  # Calculate the loss between the original and reconstructed image
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights
        if loss.item() < 1e-2:  # Stop early if the loss is small enough
            break
    return reconstructed_image

# This function handles the webcam loop, which captures frames continuously from the webcam.
# Each frame is processed by the autoencoder, saved for LLaVa interpretation, and displayed on the screen.
def webcam_loop():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam")  # Error if the webcam cannot be accessed
        return

    while not stop_event.is_set():  # Continue looping until the stop event is triggered
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Failed to capture image")  # Error if the frame is not captured
            break

        input_tensor = preprocess_frame_for_nn(frame)  # Preprocess the frame for the neural network
        processed_image = train_autoencoder(input_tensor)  # Train the autoencoder on the frame

        # Save a high-resolution version of the frame for LLaVa processing
        webcam_image_path = 'webcam_feed_image.png'
        save_high_res_image(frame, webcam_image_path)

        # Send the high-resolution image to LLaVa for real-time interpretation
        interpret_image_with_llava(webcam_image_path)

        # Display the original and processed frames
        cv2.imshow('Webcam Feed', frame)  # Display the original frame from the webcam
        processed_np = processed_image.squeeze(0).cpu().detach().numpy()  # Convert the processed image to a numpy array
        processed_np = (processed_np.transpose(1, 2, 0) * 255).astype('uint8')  # Convert the tensor to an image format
        cv2.imshow('AI Vision Memory Storage', processed_np)  # Display the reconstructed image

        # If the user presses 'q' or the stop event is set, exit the loop
        if cv2.waitKey(2000) & 0xFF == ord('q') or stop_event.is_set():
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# This class defines an AI agent that interacts using the LLaMa model.
# The agent has a name, role, backstory, and communication style, and can respond to user messages based on its visual input.
class Agent:
    def __init__(self, name, role, backstory, style, model="llama3.1:8b"):
        self.name = name
        self.role = role
        self.backstory = backstory
        self.style = style
        self.model = model
        self.conversation_history = []  # Stores the conversation history to maintain context
        self.prompt = None

    # This method creates a persona for the agent, incorporating its name, role, backstory, style, and current visual input.
    def create_prompt(self, location_description):
        persona = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Style: {self.style}\n"
            "You are in a room, aware of your surroundings through your camera eyes.\n"
            f"You can see: {location_description}\n"
            "Respond in the first person and be aware of what you observe but do not lie or add details, summarize in one sentence.\n"
        )
        return persona

    # This method generates a response from the agent based on a user message and the current location description.
    def respond(self, message, location_description):
        global llava_observation

        # Use the last observation from LLaVa instead of triggering a new one
        recent_history = self.format_conversation_history(limit=2)  # Limit to the last 2 exchanges
        prompt = self.create_prompt(location_description) + recent_history + f"User: {message}\n{self.name}:"
        
        # Add the LLaVa observation to the prompt for a context-aware response
        with llava_observation_lock:
            prompt += f" I see: {llava_observation}"

        messages = [{'role': 'user', 'content': prompt}]
        try:
            # Send the prompt to the LLaMa model and retrieve the response
            response = ollama.chat(model=self.model, messages=messages)
            response_content = response['message']['content'].strip()  # Clean up the response content
            self.conversation_history.append({'role': 'User', 'content': message})  # Store user input in history
            self.conversation_history.append({'role': self.name, 'content': response_content})  # Store agent response
            return response_content
        except Exception as e:
            return f"Error generating response: {e}"

    # This method formats the conversation history, limiting the number of exchanges included.
    def format_conversation_history(self, limit=5):
        formatted_history = ""
        for entry in self.conversation_history[-limit:]:  # Only include the last `limit` exchanges
            role = entry['role']
            content = entry['content']
            formatted_history += f"{role}: {content}\n"
        return formatted_history

# This class manages the conversation between the user and the agents.
# It allows for interactive chat, where the user can input text and the agents will respond.
class ChatManager:
    def __init__(self, agents):
        self.agents = agents

    # This method runs the interactive chat loop, allowing the user to chat with the agents.
    def interactive_chat(self):
        print("Interactive Chat is running. Type '/exit' to quit.")
        while True:
            user_input = input("You: ")  # Get user input
            if user_input.lower() in ["/exit", "/quit"]:
                stop_event.set()  # Signal the stop event to terminate both loops
                break

            main_agent = self.agents[0]

            # Wait for LLaVa to observe the environment before responding
            if not llava_has_seen.is_set():
                print("Waiting for the environment to be observed...")
                llava_has_seen.wait()  # Block until LLaVa sees

            # Main agent responds based on the LLaVa observation
            location_description = llava_observation
            response = main_agent.respond(user_input, location_description)
            print(f"{main_agent.name}: {response}")

# Initialize the agent with a name, role, backstory, and style.
agents = [
    Agent(name="LLaMa 3.1", role="Conversational AI", 
          backstory="I have advanced reasoning abilities and can 'see' through a camera.", 
          style="Conversational")
]

# Main program with stopping mechanism
def main():
    # Start the webcam loop in a separate thread
    webcam_thread = threading.Thread(target=webcam_loop)
    webcam_thread.start()

    # Start the interactive chat with the agents
    chat_manager = ChatManager(agents)
    chat_manager.interactive_chat()

    # Wait for the webcam thread to finish after stop is triggered
    webcam_thread.join()

# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import time
import threading
import ollama
import torchvision.transforms as transforms
from PIL import Image

# --- CONFIGURATION ---
# Matches the "Universal Vision Processing" section of your README
# You can swap this to "llava-phi3", "llama3.2-vision", or "gemma3:27b-it-qat"
VISION_MODEL = "gemma3:27b-it-qat" 
CHAT_MODEL = "llama3.1:8b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Starting Basic Vision Chat on {DEVICE}...")

# --- 1. CONVOLUTIONAL AUTOENCODER (Visual Memory) ---
# Matches README Section: "Convolutional Autoencoder for Visual Memory"
class FastConvAutoencoder(nn.Module):
    def __init__(self):
        super(FastConvAutoencoder, self).__init__()
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. UNIVERSAL VISION PROCESSING ---
# Matches README Section: "Universal Vision Processing (Model-Agnostic)"
def interpret_image_with_vision_model(image_path):
    try:
        res = ollama.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': 'Describe the scene in one sentence:', 'images': [image_path]}]
        )
        return res['message']['content']
    except Exception as e:
        return f"Error seeing scene: {e}"

# --- 3. MULTI-AGENT CONVERSATIONAL SYSTEM ---
# Matches README Section: "Multi-Agent Conversational System"
class Agent:
    def __init__(self):
        self.model = CHAT_MODEL

    def respond(self, message, location_description):
        prompt = f"I see: {location_description}\nUser: {message}\nAssistant:"
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error thinking: {e}"

# --- 4. REAL-TIME WEBCAM PROCESSING ---
# Orchestrates the components defined above
class VisionSystem:
    def __init__(self):
        self.current_description = "Waiting for vision..."
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.processing = False
        self.agent = Agent()
        
        # Initialize Autoencoder
        self.autoencoder = FastConvAutoencoder().to(DEVICE)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((180, 320)),
            transforms.ToTensor()
        ])

    def update_vision_thread(self, image_path):
        """Background thread to run the Vision Model (LLaVa/Gemma)"""
        desc = interpret_image_with_vision_model(image_path)
        with self.lock:
            self.current_description = desc
        self.processing = False

    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        print("📷 Webcam active. Press 'q' to quit.")
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break

            # A. Autoencoder Training (The "Neural Memory")
            input_tensor = self.transform(frame).unsqueeze(0).to(DEVICE)
            self.optimizer.zero_grad()
            recon = self.autoencoder(input_tensor)
            loss = self.criterion(recon, input_tensor)
            loss.backward()
            self.optimizer.step()

            # B. Vision Processing (The "Observer")
            # We run this async so the video feed doesn't freeze
            if not self.processing:
                self.processing = True
                cv2.imwrite("cache_vision.jpg", frame)
                threading.Thread(target=self.update_vision_thread, args=("cache_vision.jpg",)).start()

            # C. Visualization
            # Convert neural reconstruction back to image for display
            recon_img = recon.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            recon_img = (recon_img * 255).astype('uint8')
            recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)
            recon_img = cv2.resize(recon_img, (frame.shape[1], frame.shape[0]))
            
            # Show "Reality" vs "Neural Memory"
            combined = cv2.hconcat([frame, recon_img])
            cv2.imshow('Webcam Feed (Left) | Neural Autoencoder (Right)', combined)
            
            if cv2.waitKey(1) == ord('q'):
                self.stop_event.set()
        
        cap.release()
        cv2.destroyAllWindows()

# --- MAIN EXECUTION ---
def main():
    # Initialize System
    vision_sys = VisionSystem()
    
    # Start Webcam in background
    video_thread = threading.Thread(target=vision_sys.webcam_loop)
    video_thread.start()
    
    time.sleep(2) # Allow camera to warm up

    # Start Interactive Chat
    print(f"\n💬 Chat Online ({CHAT_MODEL}). Type '/quit' to exit.")
    
    while not vision_sys.stop_event.is_set():
        try:
            user_input = input("You: ")
            if user_input.lower() in ['/quit', '/exit']:
                vision_sys.stop_event.set()
                break
            
            # Get latest vision context
            with vision_sys.lock:
                current_scene = vision_sys.current_description
            
            # Generate Response
            reply = vision_sys.agent.respond(user_input, current_scene)
            print(f"AI: {reply}")
            
        except KeyboardInterrupt:
            vision_sys.stop_event.set()
            break

    video_thread.join()

if __name__ == "__main__":
    main()

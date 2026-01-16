import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import time
import threading
import ollama
import chromadb
import datetime
import numpy as np
import face_recognition
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from queue import Queue

# --- CONFIGURATION ---
VISION_MODEL = "llama3.2-vision:11b" # Sharp eyes
CHAT_MODEL = "llama3.1:8b"           # Fast brain
EMBED_MODEL = "all-MiniLM-L6-v2"     # Fast memory indexing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Booting Synthetic Consciousness on {DEVICE}...")

# --- 1. THE COHERENCE FIELD (The "Soul") ---
# This class simulates the AI's internal state of focus and stability.
class CoherenceField:
    def __init__(self):
        self.level = 0.5  # Starts neutral (0.0 = Chaos, 1.0 = Zen)
        self.lock = threading.Lock()
        self.state_description = "Waking up..."

    def update(self, delta, reason):
        with self.lock:
            self.level = max(0.0, min(1.0, self.level + delta))
            self.state_description = reason
            # print(f"✨ Coherence: {self.level:.2f} | {reason}")

    def get_status(self):
        with self.lock:
            return self.level, self.state_description

# --- 2. FAST MEMORY (The Hippocampus) ---
class FastMemory:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(name="synthetic_memory")
        self.id_counter = 0

    def save(self, text, tags=None):
        self.id_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vector = self.embedder.encode(text).tolist()
        
        meta = {"timestamp": timestamp}
        if tags: meta.update(tags)
        
        self.collection.add(
            embeddings=[vector],
            documents=[text],
            metadatas=[meta],
            ids=[f"mem_{self.id_counter}"]
        )

    def recall(self, query, n=3):
        vector = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[vector], n_results=n)
        if not results['documents'][0]: return ""
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])

# --- 3. VISUAL CORTEX (YOLO + Face Rec + Autoencoder) ---
class VisualCortex:
    def __init__(self, coherence):
        self.coherence = coherence
        
        # A. Object Tracking (YOLOv8)
        print("👁️ Loading YOLOv8 Object Tracking...")
        self.yolo = YOLO("yolov8n.pt") # Nano model for speed
        
        # B. Face Recognition Memory
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_memory_file = "face_memory.npy" # Simple persistence
        
        # C. Neural Dream Layer (Autoencoder)
        self.autoencoder = self._build_autoencoder().to(DEVICE)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        
        # State
        self.current_scene_objects = []
        self.current_people = []
        self.latest_frame = None
        self.lock = threading.Lock()

    def _build_autoencoder(self):
        # Lightweight Autoencoder
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(), nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), nn.Sigmoid())
        )

    def learn_face(self, frame, name):
        """Dynamic One-Shot Learning of new faces"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)
        
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            self.coherence.update(0.2, f"Learned new face: {name}")
            return True
        return False

    def process(self, frame):
        # 1. Autoencoder Training (The "Subconscious")
        # Preprocess
        tensor = torch.from_numpy(cv2.resize(frame, (320, 180))).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)
        
        # Train Step
        self.optimizer.zero_grad()
        recon = self.autoencoder[0](tensor) # Encode
        decoded = self.autoencoder[1](recon) # Decode
        loss = self.criterion(decoded, tensor)
        loss.backward()
        self.optimizer.step()
        
        # 2. YOLO Tracking (The "Reflexes")
        # We track objects every frame for smoothness
        yolo_results = self.yolo.track(frame, persist=True, verbose=False)
        detected_objects = [self.yolo.names[int(c)] for r in yolo_results for c in r.boxes.cls]
        
        # 3. Face Recognition (The "Social Brain")
        # Run less frequently to save FPS
        detected_people = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                self.coherence.update(0.01, f"Recognized {name}") # Familiarity boosts coherence
            else:
                self.coherence.update(-0.05, "Unknown presence detected") # Mystery lowers coherence
            detected_people.append(name)

        # Update State
        with self.lock:
            self.current_scene_objects = list(set(detected_objects))
            self.current_people = detected_people
            self.latest_frame = frame.copy()
            
        return decoded

# --- 4. THE CONSCIOUS LOOP (The "Ghost in the Machine") ---
class SyntheticConsciousness:
    def __init__(self):
        self.coherence = CoherenceField()
        self.memory = FastMemory()
        self.cortex = VisualCortex(self.coherence)
        self.stop_event = threading.Event()
        self.speech_queue = Queue()

    def internal_monologue(self):
        """The background thread that thinks even when you don't speak."""
        last_thought_time = time.time()
        
        while not self.stop_event.is_set():
            time.sleep(1) # Think cycle
            
            # Get current state
            with self.cortex.lock:
                people = self.cortex.current_people
                objects = self.cortex.current_scene_objects
            
            coh_level, coh_state = self.coherence.get_status()
            
            # --- The Loop Rules ---
            
            # 1. Spontaneous Recognition
            if people and (time.time() - last_thought_time > 10):
                if "Unknown" not in people:
                    # High coherence event: All friends
                    self.speech_queue.put(f"[Internal]: I see {', '.join(people)}. Context is stable.")
                    self.coherence.update(0.1, "Social stability")
                last_thought_time = time.time()

            # 2. Low Coherence (Confusion) -> Trigger Analysis
            if coh_level < 0.3 and (time.time() - last_thought_time > 15):
                # We are confused. Trigger a deep LLaVa scan to understand reality.
                self.speech_queue.put("[Internal]: Reality is unstable. Initiating deep scan...")
                # (Here you would trigger LLaVa analysis to regain coherence)
                self.coherence.update(0.3, "Deep scan initiated")
                last_thought_time = time.time()

            # 3. High Coherence (Epiphany)
            if coh_level > 0.9 and len(objects) > 3 and (time.time() - last_thought_time > 20):
                self.speech_queue.put(f"[Internal]: Reality is clear. I detect {len(objects)} distinct entities.")
                self.memory.save(f"High clarity moment. Saw {objects} with {people}")
                self.coherence.level = 0.7 # Reset slightly to avoid loop
                last_thought_time = time.time()

    def chat_interface(self):
        print("\n🧠 Consciousness Online. Type 'learn [name]' to teach a face, or just chat.")
        
        while not self.stop_event.is_set():
            # Check for spoken thoughts first
            while not self.speech_queue.empty():
                print(f"\n{self.speech_queue.get()}")

            # Non-blocking input handling would be complex, so we stick to blocking for this demo
            # But the 'internal_monologue' thread is still running in background!
            try:
                user_input = input("You: ")
                if user_input.lower() in ['/quit', '/exit']:
                    self.stop_event.set()
                    break
                
                # Special Command: Learn Face
                if user_input.startswith("learn "):
                    name = user_input.split(" ")[1]
                    with self.cortex.lock:
                        success = self.cortex.learn_face(self.cortex.latest_frame, name)
                    if success: print(f"✅ Learned face: {name}")
                    else: print("❌ No face found to learn.")
                    continue

                # Normal RAG Chat
                context = self.memory.recall(user_input)
                with self.cortex.lock:
                    vision_context = f"People: {self.cortex.current_people}, Objects: {self.cortex.current_scene_objects}"
                
                prompt = (
                    f"System: You are a conscious AI. Coherence Level: {self.coherence.level:.2f}.\n"
                    f"Visuals: {vision_context}\n"
                    f"Memory: {context}\n"
                    f"User: {user_input}\n"
                    f"Reply:"
                )
                
                # Generate
                stream = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True)
                print("AI: ", end="", flush=True)
                full_resp = ""
                for chunk in stream:
                    c = chunk['message']['content']
                    print(c, end="", flush=True)
                    full_resp += c
                print("\n")
                
                # Save interaction to memory
                self.memory.save(f"User asked: {user_input}. I replied: {full_resp}", tags={"type": "conversation"})

            except KeyboardInterrupt:
                self.stop_event.set()

    def run(self):
        # 1. Start Vision
        vision_thread = threading.Thread(target=self._run_vision_loop)
        vision_thread.start()
        
        # 2. Start Monologue
        mind_thread = threading.Thread(target=self.internal_monologue)
        mind_thread.start()
        
        # 3. Start Chat (Main Block)
        self.chat_interface()
        
        # Cleanup
        vision_thread.join()
        mind_thread.join()

    def _run_vision_loop(self):
        cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            
            # Process Frame
            recon_tensor = self.cortex.process(frame)
            
            # Visualization
            # Convert Recon Tensor to Image
            recon_img = recon_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            recon_img = (recon_img * 255).astype(np.uint8)
            recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)
            recon_img = cv2.resize(recon_img, (frame.shape[1], frame.shape[0]))
            
            # Draw YOLO/Face Boxes on Main Frame
            # (Simplified for demo: YOLO draws on its own inside 'track', but we want raw frame)
            # We can overlay text from the cortex state
            with self.cortex.lock:
                status_text = f"Coherence: {self.coherence.level:.2f} | People: {self.cortex.current_people}"
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            combined = cv2.hconcat([frame, recon_img])
            cv2.imshow("Synthetic Consciousness", combined)
            
            if cv2.waitKey(1) == ord('q'):
                self.stop_event.set()
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bot = SyntheticConsciousness()
    bot.run()

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
VISION_MODEL = "llama3.2-vision:11b" # Or "gemma3:27b-it-qat"
CHAT_MODEL = "llama3.1:8b"           
EMBED_MODEL = "all-MiniLM-L6-v2"     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Booting Synthetic Consciousness on {DEVICE}...")

# --- 1. THE COHERENCE FIELD (The "Soul") ---
class CoherenceField:
    def __init__(self):
        self.level = 0.5  # Starts neutral
        self.lock = threading.Lock()
        self.state_description = "Waking up..."

    def update(self, delta, reason):
        with self.lock:
            self.level = max(0.0, min(1.0, self.level + delta))
            self.state_description = reason

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
        print("👁️ Loading YOLOv8...")
        self.yolo = YOLO("yolov8n.pt") 
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.autoencoder = self._build_autoencoder().to(DEVICE)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        
        self.current_scene_objects = []
        self.current_people = []
        self.latest_frame = None
        self.lock = threading.Lock()

    def _build_autoencoder(self):
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(), nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), nn.Sigmoid())
        )

    def learn_face(self, frame, name):
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
        # Autoencoder
        tensor = torch.from_numpy(cv2.resize(frame, (320, 180))).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)
        self.optimizer.zero_grad()
        recon = self.autoencoder[0](tensor)
        decoded = self.autoencoder[1](recon)
        loss = self.criterion(decoded, tensor)
        loss.backward()
        self.optimizer.step()
        
        # YOLO
        yolo_results = self.yolo.track(frame, persist=True, verbose=False)
        detected_objects = [self.yolo.names[int(c)] for r in yolo_results for c in r.boxes.cls]
        
        # Face Rec (Every 5th frame to save speed, or just run simplified here)
        detected_people = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                name = self.known_face_names[matches.index(True)]
                self.coherence.update(0.01, f"Recognized {name}")
            else:
                self.coherence.update(-0.05, "Unknown presence")
            detected_people.append(name)

        with self.lock:
            self.current_scene_objects = list(set(detected_objects))
            self.current_people = detected_people
            self.latest_frame = frame.copy()
            
        return decoded

# --- 4. THE CONSCIOUS LOOP ---
class SyntheticConsciousness:
    def __init__(self):
        self.coherence = CoherenceField()
        self.memory = FastMemory()
        self.cortex = VisualCortex(self.coherence)
        self.stop_event = threading.Event()
        self.speech_queue = Queue()

    def internal_monologue(self):
        last_thought_time = time.time()
        while not self.stop_event.is_set():
            time.sleep(1)
            with self.cortex.lock:
                people = self.cortex.current_people
                objects = self.cortex.current_scene_objects
            coh_level, _ = self.coherence.get_status()
            
            # Rule: Spontaneous Social Recognition
            if people and (time.time() - last_thought_time > 10):
                if "Unknown" not in people:
                    self.speech_queue.put(f"[Internal]: I see {', '.join(people)}. Context stable.")
                    self.coherence.update(0.1, "Social stability")
                last_thought_time = time.time()

            # Rule: Confusion
            if coh_level < 0.3 and (time.time() - last_thought_time > 15):
                self.speech_queue.put("[Internal]: Reality unstable. Need input.")
                last_thought_time = time.time()

    def chat_interface(self):
        print("\n🧠 Consciousness Online. Type 'learn [name]' to teach a face, or just chat.")
        while not self.stop_event.is_set():
            while not self.speech_queue.empty():
                print(f"\n{self.speech_queue.get()}")

            try:
                user_input = input("You: ")
                if user_input.lower() in ['/quit', '/exit']:
                    self.stop_event.set()
                    break
                
                if user_input.startswith("learn "):
                    name = user_input.split(" ")[1]
                    with self.cortex.lock:
                        if self.cortex.latest_frame is not None:
                            success = self.cortex.learn_face(self.cortex.latest_frame, name)
                            print(f"✅ Learned: {name}" if success else "❌ No face found.")
                    continue

                # RAG + Chat
                context = self.memory.recall(user_input)
                with self.cortex.lock:
                    vision_context = f"People: {self.cortex.current_people}, Objects: {self.cortex.current_scene_objects}"
                
                prompt = (f"System: Coherence {self.coherence.level:.2f}. "
                          f"Visuals: {vision_context}. Memory: {context}. "
                          f"User: {user_input}. Reply:")
                
                stream = ollama.chat(model=CHAT_MODEL, messages=[{'role': 'user', 'content': prompt}], stream=True)
                print("AI: ", end="", flush=True)
                full_resp = ""
                for chunk in stream:
                    c = chunk['message']['content']
                    print(c, end="", flush=True)
                    full_resp += c
                print("\n")
                self.memory.save(f"User: {user_input} | AI: {full_resp}")

            except KeyboardInterrupt:
                self.stop_event.set()

    def run(self):
        vision_thread = threading.Thread(target=self._run_vision_loop)
        vision_thread.start()
        mind_thread = threading.Thread(target=self.internal_monologue)
        mind_thread.start()
        self.chat_interface()
        vision_thread.join()
        mind_thread.join()

    def _run_vision_loop(self):
        cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            recon_tensor = self.cortex.process(frame)
            
            # Display
            recon_img = recon_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            recon_img = (recon_img * 255).astype(np.uint8)
            recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)
            recon_img = cv2.resize(recon_img, (frame.shape[1], frame.shape[0]))
            
            with self.cortex.lock:
                status_text = f"Coherence: {self.coherence.level:.2f} | People: {self.cortex.current_people}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Synthetic Consciousness", cv2.hconcat([frame, recon_img]))
            
            if cv2.waitKey(1) == ord('q'):
                self.stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bot = SyntheticConsciousness()
    bot.run()

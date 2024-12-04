import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import joblib
import cv2

class PlantDiseaseDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Leaf Disease Detector - SVM")
        self.root.geometry("600x500")
        
        # Load pre-trained SVM model
        self.model = joblib.load("plant_disease_svm.pkl")  # Pre-trained SVM model
        self.class_names = ["Healthy", "Diseased"]
        
        # Create main frame with white background
        self.main_frame = tk.Frame(self.root, padx=20, pady=20, bg='white')
        self.main_frame.pack(expand=True, fill='both')
        
        # Create header
        tk.Label(self.main_frame, text="Leaf Disease Detection (SVM)", 
                 font=("Arial", 24, "bold"), bg='white').pack(pady=10)
        
        # Create upload button
        self.upload_btn = tk.Button(self.main_frame, text="Upload Leaf Image", 
                                    command=self.upload_image, 
                                    font=("Arial", 12),
                                    bg='#4CAF50', fg='white',
                                    padx=20)
        self.upload_btn.pack(pady=10)
        
        # Create image display area
        self.image_frame = tk.Frame(self.main_frame, bg='white')
        self.image_frame.pack(pady=10)
        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack()
        
        # Create analyze button
        self.analyze_btn = tk.Button(self.main_frame, text="Detect Disease", 
                                     command=self.analyze_image,
                                     font=("Arial", 12),
                                     bg='#2196F3', fg='white',
                                     state='disabled',
                                     padx=20)
        self.analyze_btn.pack(pady=10)
        
        # Create result labels
        self.status_label = tk.Label(self.main_frame, 
                                     text="", 
                                     font=("Arial", 16, "bold"),
                                     bg='white')
        self.status_label.pack(pady=5)
        
        self.percentage_label = tk.Label(self.main_frame,
                                         text="",
                                         font=("Arial", 12),
                                         bg='white')
        self.percentage_label.pack(pady=5)
        
        self.image_path = None
        self.current_image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.load_and_display_image()
            self.analyze_btn['state'] = 'normal'
            # Clear previous results
            self.status_label.config(text="")
            self.percentage_label.config(text="")

    def load_and_display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((300, 300))
        self.current_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_image)

    def analyze_image(self):
        if self.image_path:
            result = self.detect_disease(self.image_path)
            self.display_results(result)

    def detect_disease(self, img_path):
        # Extract features (color histogram)
        features = self.extract_features(img_path)
        
        # Make prediction
        prediction = self.model.predict([features])
        confidence = max(self.model.predict_proba([features])[0])  # Confidence score
        
        return {
            'status': self.class_names[prediction[0]],
            'confidence': confidence
        }

    def extract_features(self, img_path):
        # Load image with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        image = cv2.resize(image, (128, 128))
        
        # Compute color histogram features
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten the histogram
        return hist

    def display_results(self, results):
        # Display status
        status_color = "green" if results['status'] == "Healthy" else "red"
        self.status_label.config(text=f"Status: {results['status']}", fg=status_color)
        
        # Display confidence
        self.percentage_label.config(
            text=f"Confidence: {results['confidence'] * 100:.2f}%",
            fg="black"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseDetector(root)
    root.mainloop()

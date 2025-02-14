from flask import Flask, render_template, request, redirect, url_for, session, Response
from pymongo import MongoClient
from deepface import DeepFace
import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim  # SSIM for strict comparison
from model import DeepFakeDetector  # Import your deepfake detection model

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '123'

# MongoDB setup
cluster = MongoClient('mongodb://localhost:27017')
db = cluster['face']
users = db['users']

# Load deepfake detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deepfake_model = DeepFakeDetector().to(device)
deepfake_model.eval()

def detect_fake(image_path):
    """Perform deepfake detection using the pre-trained model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    output = deepfake_model(image)
    return torch.sigmoid(output).item() > 0.5  # Return True if Fake

def compute_ssim(img1_path, img2_path):
    """Compute SSIM score between two images to check similarity"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to same shape if needed
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute SSIM
    score, _ = ssim(img1, img2, full=True)
    return score

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    user = request.form['username']
    password = request.form['password']
    res = users.find_one({"username": user})
    if res and res['password'] == password:
        session['username'] = user
        return render_template('index.html', username=session['username'])
    return render_template('login.html', status='User does not exist or wrong password')

@app.route('/reg')
def reg():
    return render_template('signup.html')

@app.route('/regis', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    
    if users.find_one({"username": username}):
        return render_template('login.html', status="Username already exists")
    
    users.insert_one({"username": username, "password": password})
    return render_template('signup.html', status="Registration successful")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'fake_image' not in request.files or 'original_image' not in request.files:
            return render_template('index.html', error="Please select two files to upload.")
        
        file1 = request.files['fake_image']
        file2 = request.files['original_image']

        if file1.filename == '' or file2.filename == '':
            return render_template('index.html', error="Please select two valid files to upload.")

        fake_path = os.path.join("uploads", file1.filename)
        original_path = os.path.join("uploads", file2.filename)
        file1.save(fake_path)
        file2.save(original_path)

        # Perform deepfake detection
        is_fake = detect_fake(fake_path)

        # Perform SSIM comparison for strict verification
        ssim_score = compute_ssim(fake_path, original_path)

        # Perform face verification
        try:
            result = DeepFace.verify(img1_path=fake_path, img2_path=original_path)
            verified = result['verified']
        except Exception as e:
            return render_template('index.html', error=f"Face verification error: {str(e)}")

        # Apply stricter conditions: If SSIM < 0.5, it is Fake
        if verified and not is_fake and ssim_score >= 0.5:
            status = "Real"
        else:
            status = "Fake"

        return render_template('index.html', username=session['username'], status=status, ssim_score=ssim_score)

    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame for deepfake detection (dummy logic, replace with real model)
        # Here, we could run frame through deepfake_model
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(port=5001, debug=True)

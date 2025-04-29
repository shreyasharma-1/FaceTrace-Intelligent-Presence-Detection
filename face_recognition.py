import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd
import csv

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Paths
DATASET_DIR = 'dataset'
MODEL_FILE = 'trained_model.yml'
ATTENDANCE_DIR = 'attendance'

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

class FaceRecognition:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.dataset_dir = DATASET_DIR
        self.attendance_dir = ATTENDANCE_DIR
        self.attendance_file = os.path.join(self.attendance_dir, 'attendance.csv')
        
        # Create directories if they don't exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Initialize attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])

    def register_face(self, name, image):
        """Register a new face with the given name"""
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return False, "No face detected in the image"
            
            if len(faces) > 1:
                return False, "Multiple faces detected. Please provide an image with only one face"
            
            # Save the face image
            person_dir = os.path.join(self.dataset_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save multiple samples for better recognition
            for i, (x, y, w, h) in enumerate(faces):
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(person_dir, f'{name}_{i}.jpg'), face_img)
            
            return True, "Face registered successfully"
            
        except Exception as e:
            return False, f"Error registering face: {str(e)}"

    def train_model(self):
        """Train the face recognition model"""
        try:
            faces = []
            labels = []
            label_dict = {}
            current_label = 0
            
            # Load training data
            for person_name in os.listdir(self.dataset_dir):
                person_dir = os.path.join(self.dataset_dir, person_name)
                if os.path.isdir(person_dir):
                    label_dict[current_label] = person_name
                    for image_name in os.listdir(person_dir):
                        image_path = os.path.join(person_dir, image_name)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(image)
                        labels.append(current_label)
                    current_label += 1
            
            if not faces:
                return False, "No training data found"
            
            # Train the model
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(MODEL_FILE)
            
            # Save label dictionary
            with open('label_dict.txt', 'w') as f:
                for label, name in label_dict.items():
                    f.write(f"{label},{name}\n")
            
            return True, "Model trained successfully"
            
        except Exception as e:
            return False, f"Error training model: {str(e)}"

    def recognize_face(self, image):
        """Recognize a face in the given image"""
        try:
            # Load the model if it exists
            if not os.path.exists(MODEL_FILE):
                return None, "Model not trained yet"
            
            # Load label dictionary
            label_dict = {}
            if os.path.exists('label_dict.txt'):
                with open('label_dict.txt', 'r') as f:
                    for line in f:
                        label, name = line.strip().split(',')
                        label_dict[int(label)] = name
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None, "No face detected"
            
            # Load the trained model
            self.recognizer.read(MODEL_FILE)
            
            # Recognize the face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = self.recognizer.predict(face_roi)
                
                if confidence < 100:  # Lower confidence is better
                    name = label_dict.get(label, "Unknown")
                    return name, confidence
                
            return None, "Face not recognized"
            
        except Exception as e:
            return None, f"Error recognizing face: {str(e)}"

    def mark_attendance(self, name):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            attendance_file = os.path.join("attendance", f"{date_str}.csv")

            # Check if already marked
            if os.path.exists(attendance_file):
                with open(attendance_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith(name):
                            return True, "Already marked"

            # Mark attendance
            with open(attendance_file, "a") as f:
                f.write(f"{name},{time_str}\n")

            return True, "Attendance marked"
        except Exception as e:
            return False, str(e)


    def get_attendance(self):
        """Get attendance records"""
        try:
            records = []
            attendance_folder = "attendance"
            if not os.path.exists(attendance_folder):
                return records

            for filename in os.listdir(attendance_folder):
                if filename.endswith(".csv"):
                    date = filename.replace(".csv", "")
                    with open(os.path.join(attendance_folder, filename), "r") as f:
                        for line in f:
                            name, time = line.strip().split(",")
                            records.append({"date": date, "name": name, "time": time})
            return records
        except Exception as e:
            print("Error reading attendance:", e)
            return []

def face_extractor(img):
    """
    Detects and returns the cropped face from an image frame.
    Returns None if no face is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]


def register_face(user_id, samples=50):
    """
    Captures a given number of face samples from webcam and saves them in dataset/<user_id>/
    """
    cap = cv2.VideoCapture(0)
    count = 0
    user_path = os.path.join(DATASET_DIR, str(user_id))
    os.makedirs(user_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = face_extractor(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name = os.path.join(user_path, f"{count}.jpg")
            cv2.imwrite(file_name, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            cv2.imshow('Registering Face', face)
        if cv2.waitKey(1) == 13 or count == samples:
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    """
    Trains the LBPH face recognizer on the images in the dataset directory and saves the model.
    """
    faces = []
    labels = []

    for user_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, user_folder)
        if not os.path.isdir(folder_path):
            continue
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(np.asarray(img, dtype=np.uint8))
            labels.append(int(user_folder))

    if len(faces) == 0:
        raise RuntimeError('No face data found. Please register some faces first.')

    labels = np.asarray(labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, labels)
    model.save(MODEL_FILE)


def recognize_and_mark_attendance(threshold=70):
    """
    Runs live face recognition and marks attendance. Saves a CSV in attendance directory.
    Returns the list of attendance records.
    """
    # Load trained model
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError('Model file not found. Please train the model first.')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_FILE)

    cap = cv2.VideoCapture(0)
    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            label, confidence = model.predict(face_img)
            if confidence < threshold:
                timestamp = datetime.now()
                records.append([label,
                                timestamp.strftime('%Y-%m-%d'),
                                timestamp.strftime('%H:%M:%S')])
                cv2.putText(frame, f"ID: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Attendance - Press Enter to end', frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance to CSV
    if records:
        df = pd.DataFrame(records, columns=['UserID', 'Date', 'Time'])
        filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        filepath = os.path.join(ATTENDANCE_DIR, filename)
        df.to_csv(filepath, index=False)
    return records


if __name__ == '__main__':
    # Example usage
    # register_face(1)
    # train_model()
    # recs = recognize_and_mark_attendance()
    # print(f"Marked {len(recs)} attendance entries")
    pass

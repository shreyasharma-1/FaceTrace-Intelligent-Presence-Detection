from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import cv2
import os
import csv
from face_recognition import FaceRecognition
import numpy as np
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ATTENDANCE_FOLDER'] = 'attendance'

# Initialize face recognition
face_recognition = FaceRecognition()

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ATTENDANCE_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            if 'image' not in request.files:
                flash('No image uploaded')
                return redirect(request.url)
            
            file = request.files['image']
            if file.filename == '':
                flash('No image selected')
                return redirect(request.url)
            
            # Read image
            image_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Register face
            success, message = face_recognition.register_face(name, image)
            if success:
                flash('Face registered successfully')
                return redirect(url_for('train'))
            else:
                flash(message)
                return redirect(request.url)
                
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(request.url)
            
    return render_template('register.html')

@app.route('/train')
def train():
    try:
        success, message = face_recognition.train_model()
        flash(message)
        return render_template('train.html')
    except Exception as e:
        flash(f'Error: {str(e)}')
        return render_template('train.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        try:
            image_data = request.form['image_data']
            header, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            # Convert to OpenCV image
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Recognize and mark attendance
            name, confidence = face_recognition.recognize_face(frame)
            if name:
                success, message = face_recognition.mark_attendance(name)
                flash(f'{name}: {message}')
            else:
                flash("No face recognized.")
        except Exception as e:
            flash(f"Error: {str(e)}")

    return render_template('attendance.html')


@app.route('/view')
def view():
    records = []
    
    # Check if the attendance folder exists
    if os.path.exists(app.config['ATTENDANCE_FOLDER']):
        # Loop through each file in the attendance folder
        for filename in os.listdir(app.config['ATTENDANCE_FOLDER']):
            if filename.endswith(".csv"):  # Process only CSV files
                date = filename.replace(".csv", "")  # Get the date from the filename
                with open(os.path.join(app.config['ATTENDANCE_FOLDER'], filename), "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) == 2:  # Ensure there are exactly 2 values (name, time)
                            name, time = row  # Unpack the row into name and time
                            records.append({
                                "name": name,
                                "date": date,
                                "time": time
                            })
                        else:
                            print(f"Skipping invalid row: {row}")  # Optional: log invalid rows

    # If no records are found, show a flash message
    if not records:
        flash("No attendance records found.")
    
    
    # Pass the records to the 'view.html' template
    return render_template('view.html', records=records)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

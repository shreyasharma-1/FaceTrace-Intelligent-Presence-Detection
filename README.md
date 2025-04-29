# 🎓 Face Recognition Attendance System Web App

This is a Flask-based web application that uses OpenCV and face recognition techniques to automate attendance management using facial data.

---

## 🚀 Features

- 📝 **Face Registration** – Upload and register new user faces.
- 🧠 **Model Training** – Train the face recognition model on registered users.
- 🎥 **Live Attendance** – Capture images via webcam and mark attendance.
- 📊 **Attendance View** – View stored attendance records with sequence number, name, date, and time.
- 💾 **Persistent Storage** – Attendance is stored in CSV format for later viewing.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (custom styles), Bootstrap (optional)
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, face_recognition (dlib-based)

---

## 📂 Project Structure

```
face_attendance_app/
│
├── app.py                      # Main Flask application
├── face_recognition.py         # Face recognition logic
├── dataset/                    # Stores registered face images
├── attendance/                 # Stores attendance CSV files
├── templates/
│   ├── index.html
│   ├── register.html
│   ├── train.html
│   ├── attendance.html
│   └── view.html
└── static/
    └── styles.css              # Optional styling
```

---

## 📸 Screens & Functionalities

### ✅ Home Page
- Navigation to all core functionalities.

### 📝 Register
- Register a person’s face with their name.
- Upload image via file input.

### 🧠 Train
- Trains the model based on registered faces.
- Must be done before using the attendance feature.

### 📷 Attendance
- Takes image input via webcam (base64 encoded).
- Detects and matches face.
- Marks attendance with name, date, and time.

### 📊 View Attendance
- Displays all records in a full-screen table with:
  - Sequence number
  - Name
  - Date
  - Time

---

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask opencv-python face_recognition numpy
```

### 4. Run the App

```bash
python app.py
```

Visit: `http://localhost:5000/`

---

## 🧠 Notes

- Ensure your webcam is connected and enabled.
- Train the model after registering new faces.
- All attendance records are saved under `attendance/` as `.csv` files.


---

## 🙌 Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [face_recognition (by @ageitgey)](https://github.com/ageitgey/face_recognition)

---

## 📃 License

This project is licensed under the MIT License.

---

Let me know if you’d like to include screenshots, GitHub badges, or a demo GIF!
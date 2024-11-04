import streamlit as st
import pandas as pd
import requests
import cv2
import numpy as np
import time
import unicodedata
import re
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import os

# MongoDB configuration
client = MongoClient("mongodb+srv://cesarcorrea:k9DexhefNDS9GTLs@cluster0.rwqzs.mongodb.net/AttendanceDB?retryWrites=true&w=majority&appName=Cluster0")
db = client.AttendanceDB
students_collection = db.students
attendance_collection = db.attendance

API_URL = "https://faceappz.onrender.com/predict" 
RETRAIN_URL = "https://faceappz.onrender.com/retrain"  

DATASET_PATH = "./dataset/train/"  # Path to save student images for training

def predict_image(image):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        result = response.json()
        if "name" in result and "probability" in result:
            return result["name"], result["probability"]
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return None, None

def normalize_string(s):
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore').decode("utf-8")
    return s.lower()

def load_students_data():
    students_data = list(students_collection.find({}, {"_id": 0, "name": 1, "matricula": 1, "attendance": 1}))
    return pd.DataFrame(students_data) if students_data else pd.DataFrame(columns=["name", "matricula", "attendance"])

def load_attendance_report():
    today = datetime.now().date()
    today_datetime = datetime.combine(today, datetime.min.time())
    attendance_data = list(attendance_collection.find({"date": today_datetime}, {"_id": 0, "name": 1, "time": 1}))
    return pd.DataFrame(attendance_data) if attendance_data else pd.DataFrame(columns=["name", "time"])

def clear_attendance():
    students_collection.update_many({}, {"$set": {"attendance": False}})
    today = datetime.now()
    attendance_collection.delete_many({"date": {"$gte": today.replace(hour=0, minute=0, second=0, microsecond=0), 
                                                 "$lt": today.replace(hour=23, minute=59, second=59, microsecond=999999)}})
    st.success("Attendance has been cleared successfully.")

def save_image_to_dataset(name, image):
    """Save uploaded image to the dataset/train directory for the new student."""
    student_path = os.path.join(DATASET_PATH, name)
    os.makedirs(student_path, exist_ok=True)  # Create directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(student_path, f"{timestamp}.jpg")
    image.save(image_path)
    return image_path

def add_student(name, matricula, image):
    """Add a new student and retrain the model with their image."""
    students_collection.insert_one({"name": name, "matricula": matricula, "attendance": False})
    st.success(f"Student {name} added successfully.")
    
    # Save the image to the dataset
    image_path = save_image_to_dataset(name, image)
    
    # Trigger retraining by calling the retrain API
    response = requests.post(RETRAIN_URL)
    
    if response.status_code == 200:
        st.success("Model retrained successfully with the new student.")
    else:
        st.error("Model retraining failed.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Attendance", "Attendance Report"])

if page == "Home":
    st.image("C:/Users/cesco/Desktop/Personal/UPY/9/didier/2/proyecto/UPY Attendance Systema.jpg", use_column_width=True)
    st.write("Welcome to the UPY Attendance System. Use the sidebar to interact with the application.")

elif page == "Attendance":
    st.title("Attendance System")

    st.sidebar.subheader("Class Information")
    career = st.sidebar.selectbox("Select Career", ["Data Engineer", "Cybersecurity", "Embedded Systems", "Robotics"])
    quarter = st.sidebar.selectbox("Select Quarter", ["Immersion", "Third Quarter", "Sixth Quarter", "Ninth Quarter"])
    group = st.sidebar.selectbox("Select Group", ["A", "B"] if career == "Data Engineer" and quarter == "Ninth Quarter" else [])

    if group == "B":
        df_students = load_students_data()
        if not df_students.empty:
            st.session_state.df_students = df_students
        else:
            st.session_state.df_students = pd.DataFrame(columns=["name", "matricula", "attendance"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Data loaded successfully for group B:")
            student_table = st.empty()
            student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

        with col2:
            st.subheader("Add a new student")
            name = st.text_input("Enter Student Name")
            matricula = st.text_input("Enter Student Matricula")
            
            # Inicializar img_path para almacenar la ruta de la imagen capturada
            img_path = None
            
            if st.button("Take Photo"):
                cap = cv2.VideoCapture(0)
                st.info("Press 's' to take the photo and 'q' to quit.")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow("Press 's' to take a photo", frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        img_path = save_image_to_dataset(name, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                        st.image(img_path, caption="Captured Photo", use_column_width=True)
                        st.success("Photo taken successfully!")
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    elif cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break

            if st.button("Add Student and Retrain Model"):
                if name and matricula:
                    if img_path:
                        add_student(name, matricula, Image.open(img_path))
                        st.experimental_rerun()
                    else:
                        st.warning("Please take a photo before adding the student.")
                else:
                    st.warning("Please enter the student name and matricula.")

            if st.button("Refresh Table"):
                st.session_state.df_students = load_students_data()
                student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

            if st.button("Clear Attendance"):
                clear_attendance()
                st.session_state.df_students = load_students_data()
                student_table.dataframe(st.session_state.df_students.sort_values(by='matricula'))

        st.markdown("---")

        df_attendance = load_attendance_report()
        if not df_attendance.empty:
            st.subheader("Attendance recorded for today:")
            st.dataframe(df_attendance)
        else:
            st.write("No attendance records for today.")

elif page == "Attendance Report":
    st.title("Attendance Report")

    df_attendance_report = load_attendance_report()
    if not df_attendance_report.empty:
        st.dataframe(df_attendance_report)
    else:
        st.write("No attendance records for today.")

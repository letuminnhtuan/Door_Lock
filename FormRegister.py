import tkinter as tk
import cv2
import os
from PIL import Image, ImageTk
from tkinter import messagebox
from Handle import Create

class RegisterForm:
    def __init__(self, root, model = None, base_model = None):
        self.root = root
        self.root.title("Camera Form")
        self.create = Create(model, base_model)
        self.index = 0

        # Create the password label and entry with a larger font
        password_font = ("Arial", 14)  # Specify the desired font family and size
        self.password_label = tk.Label(self.root, text="Enter password:", font=password_font)
        self.password_label.pack()
        self.password_entry = tk.Entry(self.root, show="*", font=password_font)
        self.password_entry.pack()

        # Create the submit button
        self.submit_button = tk.Button(self.root, text="Submit", command=self.check_password)
        self.submit_button.pack()

        # Create the name label and entry (hidden initially)
        self.name_label = tk.Label(self.root, text="Enter your name:", font=("Arial", 14))
        self.name_entry = tk.Entry(self.root, font=("Arial", 14))
        self.name_label.pack_forget()
        self.name_entry.pack_forget()

        # Create the frame for displaying the camera feed (hidden initially)
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()
        self.video_frame.pack_forget()

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

    def check_password(self):
        password = self.password_entry.get()

        # Check if the password is correct (you can replace this with your own logic)
        if password == "abc":
            self.password_label.pack_forget()
            self.password_entry.pack_forget()
            self.submit_button.pack_forget()
            self.name_label.pack()
            self.name_entry.pack()
            self.video_frame.pack()

            # Create two buttons
            self.button1 = tk.Button(self.root, text="Capture", command=self.capture_picture, font=("Arial", 14))
            self.button2 = tk.Button(self.root, text="Save", command=self.save_people, font=("Arial", 14))

            # Place the buttons in the form
            self.button1.pack()
            self.button2.pack()

            # Start displaying the camera feed
            self.show_camera_feed()
        else:
            # Incorrect password message
            messagebox.showwarning("Wrong!", "Incorrect password!")

    def show_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.video_label.after(1, self.show_camera_feed)

    def capture_picture(self):
        name = self.name_entry.get()
        if len(name) == 0:
            messagebox.showwarning("Error!", "Enter your name!")
        else:
            os.makedirs(f"./Raw/{name}/", exist_ok=True)
            path = f"./Raw/{name}/img_{self.index}.jpg"
            _, img = self.cap.read()
            cv2.imwrite(path, img)
            self.index += 1

    def save_people(self):
        name = self.name_entry.get()
        if len(name) == 0:
            messagebox.showwarning("Error!", "Enter your name!")
        else:
            self.create.ProcessData(name)
            self.create.CreateEmbeddingVector(name)
            messagebox.showinfo("Done!", "Register Successful!")
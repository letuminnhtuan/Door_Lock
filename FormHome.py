import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox
import numpy as np
from FormRegister import RegisterForm
from Handle import *

# Create a class for the form
class CameraForm:
    def __init__(self, root, model, base_model, arrData):
        self.root = root
        self.root.title("Camera Form")
        self.arrData = arrData

        # Load Process
        self.process = Process(model, base_model)
        self.temp = "Tuan"
        self.count = 0

        # Create a frame to display camera feed
        self.frame = tk.Label(self.root)
        self.frame.pack()

        # Open the camera
        self.camera = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        ret, image = self.camera.read()

        # Detection
        if ret != None:
            res = self.process.model(image)
            result = res.pandas().xyxy[0]
            if (len(result)) != 0: 
                record = result.loc[0]
                xmin = np.floor(record['xmin']).astype(int)
                ymin = np.ceil(record['ymin']).astype(int)
                xmax = np.floor(record['xmax']).astype(int)
                ymax = np.ceil(record['ymax']).astype(int)
                if(record['class'] == 1):
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                elif (record['class'] == 0):
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                crop_img = image[ymin:ymax, xmin:xmax]
                resized_image = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                resized_image = np.expand_dims(resized_image, axis=0)
                vector = self.process.base_model.predict(resized_image,verbose=0)
                result = self.process.compare(vector)
                result_anti = self.process.anti_spoofing(crop_img)
                if result_anti >= 0.97:
                    cv2.putText(image, "Warning not real face!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if result != "Unknown":
                    if result == self.temp and result_anti < 0.97 and record['class'] == 0:
                        self.count += 1
                        if self.count == 10:
                            self.SendData('1')
                            print("Open")
                            self.count = 0
                    else:
                        self.count = 0
                        self.temp = result

                cv2.putText(image, result, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, str(result_anti), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to PIL Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Resize the image to fit the frame
        image = image.resize((640, 480), Image.LANCZOS)

        # Display the image on the frame
        photo = ImageTk.PhotoImage(image)
        self.frame.configure(image=photo)
        self.frame.image = photo
        self.root.after(2, self.show_frame)

    def SendData(self, data):
        self.arrData.write(data.encode('utf-8'))

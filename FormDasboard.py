from tkinter import *
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import torch
import tensorflow as tf
import serial
from FormHome import *
from FormRegister import *

class DashBoard():
    def __init__(self, root):
        self.root = root
        self.root.title("Dashboard")
        self.root.geometry("1000x400+100+50")
        self.root.resizable(False, False)
        self.root.title("Vi điều khiển")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./DetectFace.pt')
        self.base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', input_shape = (224, 224, 3), include_top=False)
        self.arrData = serial.Serial("COM4")
        # Form Dashboard
        Frame_dashboard = Frame(self.root, bg="white")
        Frame_dashboard.place(x=0, y=0, width=1000, height=400)

        title = Label(Frame_dashboard, text="Mở cửa bằng nhận diện khuôn mặt", font=("Times", 35, "bold"), fg="#6162FF", bg="white",bd=0, padx=10, pady=5, borderwidth=0)
        title.place(x=150, y=20)

        form = Label(Frame_dashboard, text="", font=("Arial", 20), bg="white", bd=2, highlightthickness=2,highlightbackground="gray").place(x=10, y=100, width=980, height=290)
        style = ttk.Style()
        style.map("TButton", foreground=[('active', '#6162FF')], background=[('active', 'gray')])
        style.configure("TButton", padding=6, relief="flat", font=("Goudy old style", 15), background="#6162FF", foreground="white")
        style.configure("TButton", borderwidth=0, focuscolor="none", highlightthickness=0, bordercolor="none")
        style.configure("TButton", bordercolor="none", focuscolor="none", borderwidth=0)
        style.configure("TButton", bordercolor="none", focuscolor="none", borderwidth=0, relief="flat", background="#6162FF", foreground="white", padding=6)
        style.configure("TButton", bordercolor="none", focuscolor="none", borderwidth=0, relief="flat", background="#6162FF", foreground="orange", padding=6, borderradius=80)
        # Button quản lí doanh thu
        ttk.Button(Frame_dashboard, text="Open", command=self.transfer_Open_Door).place(x=20, y=110,width=160, height=40)
        # Button đăng kí xe
        ttk.Button(Frame_dashboard, text="Register", command=self.transfer_Regist).place(x=430, y=110, width=160, height=40)
        # Out
        ttk.Button(Frame_dashboard, text="Out", command=self.out).place(x=820, y=110, width=160, height=40)


    # Form Open
    def transfer_Open_Door(self):
        self.open_Open_Door()
    def open_Open_Door(self):
        self.new_window = Toplevel(self.root)
        self.app = CameraForm(self.new_window, self.model, self.base_model, self.arrData)
    
    # Form Register
    def transfer_Regist(self):
        self.open_Regist()
    def open_Regist(self):
        self.new_window = Toplevel(self.root)
        self.app = RegisterForm(self.new_window, self.model, self.base_model)

    def out(self):
        self.root.destroy()

root = Tk()
obj = DashBoard(root)
root.mainloop()
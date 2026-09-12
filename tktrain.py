import subprocess as sb
import tkinter as tk
import sys
import os
import threading

from constants import (
    PROJECT_ROOT
)

root = tk.Tk()
root.title("Training")
root.config(bg="black")

def start_training():
    def x():
        path = os.path.join(PROJECT_ROOT, "train.py")
        sb.run([sys.executable, path])
    threading.Thread(target=x, daemon=True).start()
    label.config(text="Your AI is being trained. You can return to the navigation menu.")

tk.Label(root,
         text="Training",
         bg="black",
         fg="Green",
         font=("Courier", 20, "bold"),).pack()

button = tk.Button(root,
                   text="Start Training",
                   command=start_training,
                   bg="green",
                   fg="white",
                   activebackground="black",
                   activeforeground="white",
                   highlightbackground="black",
                   highlightcolor="black",
                   borderwidth=1,
                   font=("Courier", 12))
button.pack()

label = tk.Label(root,
                 bg="black",
                 fg="Green",
                 font=("Courier", 12),
                 text="",)
label.pack()

root.mainloop()
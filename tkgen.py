import tkinter as tk
import subprocess as sb
import os
import sys
import threading

from constants import (
    PROJECT_ROOT
)
import generate

from tkinter import messagebox as msgbox

root = tk.Tk()
root.config(bg="black")
root.title("Video Generator")

output_str = tk.StringVar()
output_str.set(os.path.join(PROJECT_ROOT, 'outputs'))

output_str2 = tk.StringVar()
output_str2.set('30')

resolution_str = tk.StringVar()
resolution_str.set('1080p')

Resolution = tk.StringVar()
Resolution.set('Choose Resolution')

def gen_video():
    def main():
        if Resolution.get() != 'Choose Resolution' and prompt:
            path = os.path.join(PROJECT_ROOT, "generate.py")
            output_ = generate.generate(prompt, output=output, steps=steps, resolution=resolution)
            msgbox.showinfo("Video Generated at ", output_)
            path_text.config(text='Path: ' + output_)
            path_text.pack()

tk.Label(root, text="Video Generator (From Scratch)", font=("Courier", 20), bg='black', fg='white').pack()

# ========================================================================================================

p_ef = tk.Frame(root, bg="black")
p_ef.pack()

tk.Label(p_ef, text="Prompt: ", bg="black", fg="green", font=("Courier", 12)).pack(side=tk.LEFT)

PromptEntry = tk.Entry(p_ef, bg="black", fg="green", font=("Courier", 12))
PromptEntry.pack(side=tk.LEFT)

prompt = PromptEntry.get()

# ========================================================================================================

o_ef = tk.Frame(root, bg="black")
o_ef.pack()

tk.Label(o_ef, text='Output: ', bg="black", fg="green", font=('Courier', 12)).pack(side=tk.LEFT)

OutputEntry = tk.Entry(o_ef, bg="black", fg="green", textvariable=output_str, font=('Courier', 12))
OutputEntry.pack(side=tk.LEFT)

output = OutputEntry.get()

# ========================================================================================================

s_ef = tk.Frame(root, bg="black")
s_ef.pack()

tk.Label(s_ef, text='Steps: ', bg="black", fg="green", font=('Courier', 12)).pack(side=tk.LEFT)

StepsEntry = tk.Entry(s_ef, bg="black", fg="green", textvariable=output_str2, font=('Courier', 12))
StepsEntry.pack(side=tk.LEFT)

steps = StepsEntry.get()

# ========================================================================================================

r_ef = tk.Frame(root, bg="black")
r_ef.pack()

tk.Label(r_ef, text="Resolution: ", bg='black', fg="green", font=('Courier', 12)).pack(side=tk.LEFT)

Options = tk.OptionMenu(
    r_ef,
    Resolution,
    "1080p",
    "720p",
    "480p"
)

Options.config(bg="black", fg="green", font=('Courier', 12))
Options.pack(side=tk.LEFT)

resolution = Resolution.get()

# ========================================================================================================

startbutton = tk.Button(root, text="Start", bg="green", fg="black", font=('Courier', 12), command=gen_video)
startbutton.pack()

path_text = tk.Label(bg="black", fg="green", font=('Courier', 12), text='')
path_text.pack()

# ========================================================================================================

root.mainloop()
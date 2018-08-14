import tkinter as tk

root = tk.Tk()

top_frame = tk.Frame(root)
top_frame.pack()
bot_frame = tk.Frame(root)
bot_frame.pack(side='bottom')

label = tk.Label(top_frame, text='hello world, from tkinter')
label.pack()

button = tk.Button(top_frame, text='click me!', fg='blue', bg='orange')
button.pack(fill='x')
button2 = tk.Button(None, text='no, click me!', fg='orange', bg='blue')
button2.pack(side='left', fill='y')
button3 = tk.Button(None, text='ignore them, click me!', fg='orange', bg='blue')
button3.pack(side='right', fill='y')

root.mainloop()
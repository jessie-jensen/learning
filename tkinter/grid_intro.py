import tkinter as tk

root = tk.Tk()

label = tk.Label(root, text='Name: ')
label.grid(row=0, column=0)

entry = tk.Entry(root)
entry.grid(row=0, column=1)

cbutton = tk.Checkbutton(root, text='Remember name')
cbutton.grid(row=1, columnspan=2)

button = tk.Button(root, text='OK')
button.bind('<Button-1>', print('**CLICK**'))
button.grid(row=2, columnspan=2)

root.mainloop()
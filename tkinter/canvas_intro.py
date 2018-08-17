import tkinter as tk

root = tk.Tk()

canvas = tk.Canvas(root, width = 300, height = 300, bg='grey')
canvas.pack()

canvas.create_rectangle(20,20,200,150, outline='orange')
canvas.create_rectangle(160,110,260,210, outline='orange')

canvas.create_line(0,0,299,299, fill='orange')

canvas.create_arc(0,250,299,299, extent=180, style='arc', outline='orange')

canvas.create_text(150,230, text='Hello World Inc.', font=('Comic Sans MS', 30, 'bold'), fill='orange')

root.mainloop()
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()

ans = messagebox.askquestion('Q1', 'foo?')
print(ans)

if ans=='yes':
    messagebox.showinfo('A1', 'bar')
elif ans=='no':
    messagebox.showinfo('A1', 'baz')

root.mainloop()
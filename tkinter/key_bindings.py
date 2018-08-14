import tkinter as tk

root = tk.Tk()

def key(event):
    print('pressed', repr(event.char))

def callback(event):
    frame.focus_set()
    mouse_buttons = {1:'left', 2:'right'}
    print(event)
    print(mouse_buttons[event.num], 'clicked at', event.x, event.y)

def evaluate(event):
    s = e.get()
    try:
        ans.configure(text='Answer = ' + str(eval(s)))
    except:
        ans.configure(text=s)

frame = tk.Frame(root, width=100, height=100)
frame.bind('<Key>', key)
frame.bind('<ButtonPress>', callback)
frame.pack()

e = tk.Entry(root)
e.bind('<Return>', evaluate)
e.pack()
e.focus_set()

ans = tk.Label(root)
ans.pack()

root.mainloop()
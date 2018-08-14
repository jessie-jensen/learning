import tkinter as tk

root = tk.Tk()

equation_s = ''
equation_sv = tk.StringVar()
equation_sv.set(equation_s)
equation_label = tk.Label(root, textvariable=equation_sv, fg='grey')
equation_label.grid(row=0, columnspan=4)

total_s = ''
total_sv = tk.StringVar()
total_sv.set(total_s)
total_label = tk.Label(root, textvariable=total_sv, font=('Ariel Black',16))
total_label.grid(row=1, columnspan=4)



def btnPress(x):
    global equation_s

    if x=='C':
        equation_s = ''
        equation_sv.set(equation_s)
        total_s = ''
        total_sv.set(total_s)
    elif x=='=':
        total_s = str(eval(equation_s))
        total_sv.set(total_s)
        equation_s = total_s
    else:
        equation_s += str(x)
        equation_sv.set(equation_s)



buttons_l = ['C',None,None,'+',1,2,3,'-',4,5,6,'*',7,8,9,'/',None,'.',0,'=']
for i in range(len(buttons_l)):
    if buttons_l[i] != None:
        tk.Button(root, text='  '+str(buttons_l[i])+'  ', command=lambda i=i: btnPress(buttons_l[i]))\
            .grid(row=(2+int(i/4)), column=i%4)



root.mainloop()
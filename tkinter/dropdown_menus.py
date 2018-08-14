import tkinter as tk

root = tk.Tk()

def menu_func():
    print('this is a menu func')

main_menu = tk.Menu(root)
root.configure(menu = main_menu)

sub_menu = tk.Menu(main_menu)
main_menu.add_cascade(label='foo', menu=sub_menu)
sub_menu.add_command(label='bar', command=menu_func)
sub_menu.add_separator()
sub_menu.add_command(label='baz', command=menu_func)

root.mainloop()
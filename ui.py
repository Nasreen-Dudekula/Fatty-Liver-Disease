from tkinter import *
from tkinter.ttk import *
import os


# Top level window 
root = Tk()

root.title("Prediction Of Fatty Liver Disease Using Machine Learning Algorithms")
root.geometry('600x300')

var =  IntVar()

def sel():  
    print('')

style = Style(root) 
style.configure("TRadiobutton", background = "light green",  
                foreground = "red", font = ("arial", 10, "bold")) 

ans1 = Radiobutton(root, text='NB-Existing System', variable=var, value=1,command=sel)
ans2 = Radiobutton(root, text='RF-Proposed System', variable=var, value=2,command=sel)
ans3 = Radiobutton(root, text='XGB-Enhanced System', variable=var, value=3,command=sel)
ans4 = Radiobutton(root, text='Quit', variable=var, value=4,command=sel)

ans1.pack()
ans2.pack()
ans3.pack()
ans4.pack()

def out():    
    global ans1, ans3, ans3 ,ans4    
    answer = (ans1 or ans2 or ans3 or ans4(var.get()))
    system = (var.get())
    if system==1:
        os.system('python es.py')
    if system==2:        
        os.system('python ps.py')        
    if system==3:        
        os.system('python enh.py')        
    if system==4:
        root.destroy()

button = Button(root,text = "Proceed",command = out)
button.pack()
root.mainloop()
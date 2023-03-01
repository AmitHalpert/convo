from tkinter import *
from tkinter import ttk
import dataCollection
import handSignDetection

root = Tk()
frm = ttk.Frame(root, padding=10)
root.geometry('320x200')
root.resizable(False, False)
root.title('convo')
frm.grid()
ttk.Button(frm, text="Start data collection", command=dataCollection.start).grid(column=1, row=0)
ttk.Button(frm, text="Start and Sign Detection", command=handSignDetection.start).grid(column=2, row=0)

root.mainloop()

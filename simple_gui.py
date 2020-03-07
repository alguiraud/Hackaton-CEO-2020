from tkinter import Tk, Label, Button, Entry
from tkinter import filedialog as tkfd
import new_object_detect as ov

class Application:
	def __init__(self, master, device):
		self.master = master
		master.title("Welcome to the Eye-C demonstrator")

		self.imager = ov.InferImage(device)
	
		self.lbl = Label(master, text="Please choose an image", font=("Arial Bold", 25))
		self.lbl.grid(column=1, row=0)

		self.initial = "/"

		self.btn = Button(master, text="Choose image...", command=self.clicked)
		self.btn.grid(column=1, row=2)
		self.path = ""

	def clicked(self):
		entry = Entry(self.master)
		self.path = tkfd.askopenfilename(initialdir = self.initial, title = "Select file",
                    		filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		self.initial = self.path.split()[:-2]
		
		if self.path != "": self.imager.doit(self.path)

if __name__ == '__main__':
	window = Tk()
	app = Application(window)
	window.mainloop()
from tkinter import Tk
import simple_gui as gui
import sys

if __name__ == "__main__":
	window = Tk()
	app = gui.Application(window, sys.argv[1])
	window.mainloop()

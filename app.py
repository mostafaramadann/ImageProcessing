"""
Author: Mostafa Ramadan 
Start Date: 20th August 10:30 PM
"""
from tkinter import *
from tkinter import filedialog as fd 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from process import * ### image processing file

class Window():
    
    """
    This Class Represents a Tkinter Window
    """
   

    def __init__(self,title="Image Project",dimensions="1366x768",\
                            bg="white"):
       
        self.text_widgets=list()
        self.radio_buttons = list()
        self._window = Tk()
        self._window.configure(background=bg)
        self._window.title(title)

        self._window.geometry(dimensions)

        self.add_button(color="green",text="Apply",x=1150,y=400,\
                         binding=self.apply)

        self.add_button(color="red",text="Reset",x=1220,y=400,\
                         binding=self.reset)
        self.add_button(color="red",text="Clear Cache",x=1170,y=435,\
                    binding=self.clear)

        self.add_button(color="gray",text="open",x=1150,y=650,\
        binding= self.openfile)

        ## radio button variable        
        self.r_var = IntVar(self._window)
        self.r_var.set(0)
        x_button = 1150
        self.add_radiobutton(x=x_button,y=25,text="Blur")

        self.add_radiobutton(x=x_button,y=50,text="Gray")

        self.add_radiobutton(x=x_button,y=75,text="LeftSobel")

        self.add_radiobutton(x=x_button,y=100,text="RightSobel")

        self.add_radiobutton(x=x_button,y=125,text="TopSobel")

        self.add_radiobutton(x=x_button,y=150,text="BottomSobel")

        self.add_radiobutton(x=x_button,y=175,text="Sharp")

        self.add_radiobutton(x=x_button,y=200,text="Unsharp")

        self.add_radiobutton(x=x_button,y=225,text="Noise")

        self.add_radiobutton(x=x_button,y=250,text="Sepia")

        self.add_radiobutton(x=x_button,y=275,text="FullSobel")

        self.add_radiobutton(x=x_button,y=300,text="Special1")

        self.add_radiobutton(x=x_button,y=325,text="Special2")

        self.add_radiobutton(x=x_button,y=350,text="equalize")
        
        self.add_radiobutton(x=x_button,y=375,text="rotate")

        self.add_radiobutton(x=x_button-150,y=25,text="Shear")
        self.add_radiobutton(x=x_button-150,y=50,text="apply brightness")

        self.add_radiobutton(x=x_button-150,y=75,text="emboss")
        self.add_radiobutton(x=x_button-150,y=100,text="outline")

        self.gray = None ### Caches
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.full_sobel = None
        self.noise = None
        self.blur = None

        self.file=None

        self.add_text(x=x_button,y=460,width=18,height=1)
        self.add_text(x=x_button,y=480,width=18,height=1)
        self.add_text(x=x_button,y=500,width=18,height=1)


    def start(self)-> None:
        """ Starts the window Loop """
        self._window.mainloop()
    
    def add_canvas(self,x=0,y=0):
        f = Figure(figsize=(9,9))
        self.a = f.add_subplot(2,1,1)
        self.a.axis("off")
        self.a.imshow(self.image,cmap="gray")
        self.a2 = f.add_subplot(2,1,2)
        self.a2.axis("off")
        self.a2.imshow(self.cimage,cmap="gray")
        self.canvas = FigureCanvasTkAgg(f, master=self._window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas._tkcanvas.place(x=x,y=y)

    def add_button(self,color="gray",binding=None,text="button",x=0,y=0):

        button = Button(self._window,text=text,command=lambda:binding())
        button.configure(background=color)
        button.pack()

        button.place(x=x,y=y)

    def add_text(self,color="#edebe6",x=10,y=25,height=6,width=50):

        text_widget = Text(self._window,height=height,width=width)
        text_widget.configure(background=color)
        text_widget.pack()

        text_widget.place(x=x,y=y)
        self.text_widgets.append(text_widget)

    def add_radiobutton(self,x=10,y=25,text="r_button"):

        r_button = Radiobutton(self._window,text=text,padx=20,\
                                value=len(self.radio_buttons),width=10\
                                ,variable=self.r_var)

        r_button.pack()
        r_button.place(x=x,y=y)

        self.radio_buttons.append(r_button)

    def openfile(self):
        self.file = fd.askopenfilename()
        self.image = plt.imread(self.file)
        self.cimage = self.image.copy()
        self.add_canvas(x=-20,y=-100)

    def apply(self):

        if self.r_var.get()  == 0:
            if self.blur is None:
                self.cimage = median_blur(self.cimage)
                self.cimage = applykernel(self.cimage,"blur",(3,3))
                self.cimage = avg_blur(self.cimage)
                self.blur = self.cimage
            else:
                self.cimage = self.blur
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()   

        elif self.r_var.get() == 1:

            if len(self.cimage.shape)!=3:
                self.gray = self.cimage
            elif self.gray is None:
                rgb_weights = [0.2989, 0.5870, 0.1140]
                self.cimage = color_filter(self.cimage,rgb_weights,gray=True)
                self.gray = self.cimage.copy()  
            else:
                self.cimage = self.gray ## cache

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()
               

        elif self.r_var.get() == 2:

            if self.left is None:
                self.cimage = applykernel(self.cimage,"left_sobel",(5,5))
                self.left = self.cimage.copy()
            else:
                self.cimage = self.left

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()  

        elif self.r_var.get() == 3:
            if self.right is None:   
                self.cimage = applykernel(self.cimage,"right_sobel",(5,5))
                self.right = self.cimage.copy()
            else:
                self.cimage = self.right

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 4:
            if self.top is None:
                self.cimage = applykernel(self.cimage,"top_sobel",(5,5))
                self.top = self.cimage.copy()
            else:
                self.cimage = self.top
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw() 

        elif self.r_var.get() == 5:
            if self.bottom is None:
                self.cimage = applykernel(self.cimage,"bottom_sobel",(5,5))
                self.bottom = self.cimage.copy()
            else:
                self.cimage = self.bottom

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw() 

        elif self.r_var.get() == 6:
            self.cimage = applykernel(self.cimage,"sharpen",(5,5))
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw() 

        elif self.r_var.get() == 7:
            self.cimage = applykernel(self.cimage,"unsharp")
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 8:
            if self.noise is None:
                self.cimage = applykernel(self.cimage,"noise",(3,3))
                self.noise = self.cimage
            else:
                self.cimage = self.noise
                
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw() 
        elif self.r_var.get() == 9:
            self.cimage = applykernel(self.cimage,"sepia",(3,3))
            self.a2.imshow(self.cimage)
            self.canvas.draw() 

        elif self.r_var.get() == 10:
            if self.left is not None and self.right is not None\
            and self.top is not None and self.bottom is not None and self.gray is not None:

                self.cimage = self.gray.copy()
                self.cimage = self.top+self.bottom+self.left+self.right
                self.a2.imshow(self.cimage,cmap="gray")
                self.canvas.draw()
                self.full_sobel = self.cimage

        elif self.r_var.get() == 11:
            if self.full_sobel is not None:
                self.cimage = self.image.copy().astype("float64")
                for i in range(3):
                    self.cimage[:,:,i]+=self.full_sobel.astype("float64")

                n = self.cimage-self.cimage.min()
                d = self.cimage.max()-self.cimage.min()
                self.cimage = (n/d)*1.5
                self.a2.imshow(self.cimage)
                self.canvas.draw()

        elif self.r_var.get() == 12:
            text = self.text_widgets[1].get("1.0","end")
            weights = text.split(",")
            weights = [float(w) for w in weights]
            self.cimage = self.image.copy()
            self.cimage = color_filter(self.cimage,weights)/255

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()
        elif self.r_var.get() == 13:
            self.cimage = equalize(self.cimage)

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 14:
            if self.gray is not None:
                self.cimage = self.gray
                rotation_angle = self.text_widgets[0].get("1.0","end")
                rotation_angle = int(rotation_angle) if rotation_angle!="\n" else 0
                if rotation_angle >0 and rotation_angle<360:
                    self.cimage = rotate(self.cimage,rotation_angle)

            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 15:
            self.cimage = shear(self.cimage)
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 16:
            value = self.text_widgets[2].get("1.0","end")
            self.cimage = self.cimage*float(value)
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 17:
            self.cimage = applykernel(self.cimage,"emboss",(3,3))
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw()

        elif self.r_var.get() == 18:
            self.cimage = applykernel(self.cimage,"outline",(3,3))
            self.a2.imshow(self.cimage,cmap="gray")
            self.canvas.draw() 


            
           

    def reset(self):
        self.cimage = self.image
        self.a2.imshow(self.cimage,cmap="gray")
        self.canvas.draw() 

    def clear(self):
        self.gray = None
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.full_sobel = None
        self.noise = None
        self.blur = None
        self.cimage = self.image


if __name__ == "__main__":
    window = Window()
    window.start()
import numpy as np
import math
from random import Random

def convolute(im,kernel):
    """
    convolution with one or more number of channels
    """
    image = im.copy()
    height,width = image.shape[0],image.shape[1]
    channels = None
    
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    if len(image.shape) == 3:
        channels = image.shape[2]
        
    for row in range(height):
        for column in range(width):
            if row+kernel_height<height and column+kernel_width<width:
                    image[row,column] = np.sum(image[row:row+kernel_height, column:column+kernel_width]*kernel)

                    if image[row,column]>255:
                        image[row,column]=255
                    elif image[row,column]<0:
                        image[row,column]=0
            
    return image

def color_filter(image,weights,gray=False):
    
    height,width = image.shape[0],image.shape[1]
    channels = image.shape[2]
    
    if gray:
        ret = np.zeros((height,width))
    else:
        ret = np.zeros(image.shape)
    for row in range(height):
        for column in range(width):
            if gray:
                ret[row,column]= np.sum(image[row,column]*weights) ## red would get mul with red_weight etc.
            else:
                for channel in range(channels):
                    ret[row,column,channel]=image[row,column,channel]*weights[channel]
                    if ret[row,column,channel]>255:
                        ret[row,column,channel] = 255
    return ret

def rotate(image,angle):
    rad = angle*(math.pi/180)
    rotation_matrix=np.array([
        
        [ math.cos(rad), -math.sin(rad) ],
        [ math.sin(rad),  math.cos(rad) ]
    ])
    
    height,width = image.shape[0],image.shape[1]
    
    dim = max(height,width)
    ret = np.zeros((dim*2,dim*2))
    
    for row in range(height):
        for column in range(width):
            
            coords = np.array([column+dim/2,row+dim/2])
            new_coords = np.dot(coords,rotation_matrix)
            
            x,y = int(round(new_coords[0]-dim/4)),int(round(new_coords[1]-dim/4))
            ret[y,x] = image[row,column]
    
    return ret

def shear(image):
    shear_matrix_X = np.identity(2)
    shear_matrix_Y = np.identity(2)
    shear_matrix_X[0,1] = 0.1
    shear_matrix_Y[1,0] = 0.2

    height,width = image.shape[0],image.shape[1]
    dim = max(height,width)
    ret = np.zeros((int(dim*1.2),int(dim*1.2)))


    for row in range(height):
        for column in range(width):
            coords = np.array([column,row])
            x = int(np.dot(shear_matrix_X,coords)[0])
            y = int(np.dot(shear_matrix_X,coords)[1])
            nx = x+int(dim/8)
            ny = y+int(dim/6)
            if nx<int(dim*1.2) and ny<int(dim*1.2):
                ret[ny,nx] = image[row,column]
    return ret


def get_kernel(kernel_name,kernel_shape=(3,3)):
    kernel = None
    
    if kernel_name == "blur":
        kernel = np.ones(kernel_shape)/8
        kernel[0,0] = 0.0625
        kernel[0,-1] = 0.0625
        kernel[-1,0] = 0.0625
        kernel[-1,-1] = 0.0625
        kernel[1,1] = 0.25
    elif kernel_name == "sharpen":
        kernel = np.zeros(kernel_shape)
        mid  = int(3/2)
        kernel[mid,mid] = 5
        kernel[mid,mid-1] = -1
        kernel[mid,mid+1] = -1
        kernel[mid-1,mid] = -1
        kernel[mid+1,mid] = -1
    elif kernel_name == "left_sobel":
        kernel = np.zeros(kernel_shape)
        kernel[0,0] = 1
        kernel[1,0] = 2
        kernel[2,0] = 1
        kernel[0,-1] = -1
        kernel[1,-1] = -2
        kernel[2,-1] = -1
    elif kernel_name == "right_sobel":
        kernel = np.zeros(kernel_shape)
        kernel[0,0] = -1
        kernel[1,0] = -2
        kernel[2,0] = -1
        kernel[0,-1] = 1
        kernel[1,-1] = 2
        kernel[2,-1] = 1
    elif kernel_name == "top_sobel":
        kernel = np.zeros(kernel_shape)
        kernel[0,0] = 1
        kernel[0,1] = 2
        kernel[0,2] = 1
        kernel[-1,0] = -1
        kernel[-1,1] = -2
        kernel[-1,2] = -1
    elif kernel_name == "bottom_sobel":
        kernel = np.zeros(kernel_shape)
        kernel[0,0] = -1
        kernel[0,1] = -2
        kernel[0,2] = -1
        kernel[-1,0] = 1
        kernel[-1,1] = 2
        kernel[-1,2] = 1
    elif kernel_name == "sepia":
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])
    elif kernel_name == "emboss":
        kernel = np.array([[-2,-1,0],
                            [-1,1,1],
                            [0,1,2]])
                            
    elif kernel_name == "outline":
        kernel = np.array([[-1,-1,-1],
                           [-1,8,-1],
                           [-1,-1,-1]])
    return kernel

def blur(image,kernel_shape):
    kernel = get_kernel("blur",kernel_shape)
    return convolute(image,kernel)

def median_blur(image):

    h,w = image.shape[0],image.shape[1]
    new_image = image.copy()

    def sort(kernel):
        median_filter = []
        for row in range(3):
            for column in range(3):
                median_filter.append(kernel[row,column])

        #### bubble sort ####
        for i in range(9):
            for k in range(0,8-i):
                if  median_filter[k+1] < median_filter[k]:
                    ### Swap statement ###
                    median_filter[k+1], median_filter[k] = median_filter[k], median_filter[k+1]
                    
        return median_filter[4]

    for row in range(h):
        for column in range(w):
            if row+3<h and column+3<w:
                part = image[row:row+3,column:column+3]
                kernel = part.copy()
                median = sort(kernel)

                new_image[row:row+3,column:column+3] = median
    return new_image

def avg_blur(image,x=1):
    h,w = image.shape[0],image.shape[1]
    new_image = image.copy()
    for _ in range(x):
        for row in range(h):
            for column in range(w):
                if row+3<h and column+3<w:
                    part = new_image[row:row+3,column:column+3] 
                    new_image[row,column] = np.sum(part)/9
    return new_image

def sharpen(image,kernel_shape):
    
    kernel = get_kernel("sharpen",kernel_shape)
    return convolute(image,kernel)

def unsharp_highboost(image,kernel_shape):
    kernel = get_kernel("blur",kernel_shape)
    blurred = convolute(image,kernel)
    mask = image - blurred
    return image+mask*1

def sobel(image,sobel_type,kernel_shape):
    kernel = get_kernel(sobel_type,kernel_shape)
    return convolute(image,kernel)

def emboss(image):
    kernel = get_kernel("emboss")
    return convolute(image,kernel)

def outline(image):
    kernel = get_kernel("outline")
    return convolute(image,kernel)

def noise(image):
    ret = image.copy()
    height,width= image.shape[0],image.shape[1]
    
    for row in range(height):
        for column in range(width):
            noise = int(round(Random().random()+0.45))
            if noise>1:
                noise=1
            ret[row,column] *=noise 
            
    return ret

def sepia(image):
    kernel = get_kernel("sepia")
    image = image.copy().astype("float64")
    height,width = image.shape[0],image.shape[1]

    for row in range(height):
        for column in range(width):
                r,g,b = image[row,column]
                colors = []
                for roww in kernel:
                    color = int(r*roww[0]+g*roww[1]+b*roww[2])
                    if color>255:
                        color = 255
                    colors.append(color)
                image[row,column] = colors

    image = (image-image.min())/(image.max()-image.min())
    return image


def applykernel(image,kernel_name,kernel_shape=(3,3)):
    
    if kernel_name == "blur":
        image = blur(image,kernel_shape)
    elif kernel_name == "sharpen":
        image = sharpen(image,kernel_shape)
    elif kernel_name == "unsharp":
        image = unsharp_highboost(image,kernel_shape)
    elif kernel_name == "left_sobel":
        image = sobel(image,kernel_name,kernel_shape)
    elif kernel_name == "right_sobel":
        image = sobel(image,kernel_name,kernel_shape)
    elif kernel_name == "top_sobel":
        image = sobel(image,kernel_name,kernel_shape)
    elif kernel_name == "bottom_sobel":
        image = sobel(image,kernel_name,kernel_shape)
    elif kernel_name == "noise":
        image = noise(image)
    elif kernel_name == "sepia":
        image = sepia(image)
    elif kernel_name == "emboss":
        image = emboss(image)
    elif kernel_name == "outline":
        image = outline(image)

    return image

def equalize(image):
    height,width = image.shape[0],image.shape[1]
    frequencies = np.array([0 for i in range(0,256)])

    ret = np.zeros((height,width))

    for row in range(height):
        for column in range(width):
            pixel_value = image[row,column]
            frequencies[int(pixel_value)]+=1

    frequencies = frequencies / (height*width)

    values = [0 for i in range(256)]
    nums = list()

    for i in range(frequencies.shape[0]):
        nums.append(frequencies[i])
        values[i] = 255*np.sum(np.sum(np.array(nums))) # max * prob = value of pixel

    values = np.array(values)
    for row in range(height):
        for column in range(width):
            index = image[row,column]
            ret[row,column] = values[int(index)]

    return ret



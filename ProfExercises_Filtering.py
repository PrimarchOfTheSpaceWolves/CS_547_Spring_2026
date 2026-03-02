###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn
import timm
import torchvision
from enum import Enum
from torch import nn
from torchvision.transforms import v2
from numpy.lib.stride_tricks import sliding_window_view

class FilterType(Enum):
    BOX = "Box Filter"
    GAUSS = "Gaussian Filter"
    MEDIAN = "Median Filter"
    LAPLACE = "Laplacian Filter"
    LAP_SHARP = "Laplacian Sharpening"
    SOBEL_X = "Sobel in X"
    SOBEL_Y = "Sobel in Y"
    GRAD_MAG = "Gradient Magnitude"
    
def do_filter(image, filter_size, filter_type):
    if filter_type == FilterType.BOX:
        output = cv2.blur(image, ksize=(filter_size, filter_size))
    elif filter_type == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, 
                                  ksize=(filter_size, filter_size),
                                  sigmaX=0)
    elif filter_type == FilterType.MEDIAN:
        output = cv2.medianBlur(image, ksize=filter_size)
    elif filter_type == FilterType.LAPLACE:
        laplace = cv2.Laplacian(image, 
                                ddepth=cv2.CV_64F, 
                                ksize=filter_size,
                                scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127)
    elif filter_type == FilterType.LAP_SHARP:
        laplace = cv2.Laplacian(image, 
                                ddepth=cv2.CV_64F, 
                                ksize=filter_size,
                                scale=0.25)
        fimage = image.astype("float64")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
    elif filter_type == FilterType.SOBEL_X:
        sbx = cv2.Sobel(image, 
                        ddepth=cv2.CV_64F, 
                        dx=1, dy=0, 
                        ksize=filter_size, 
                        scale=0.25)
        output = cv2.convertScaleAbs(sbx, alpha=0.5, beta=127)
    elif filter_type == FilterType.SOBEL_Y:
        sby = cv2.Sobel(image, 
                        ddepth=cv2.CV_64F, 
                        dx=0, dy=1, 
                        ksize=filter_size, 
                        scale=0.25)
        output = cv2.convertScaleAbs(sby, alpha=0.5, beta=127)
    elif filter_type == FilterType.GRAD_MAG:
        sbx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=filter_size, scale=0.25)
        sby = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=filter_size, scale=0.25)
        grad_mag = np.absolute(sbx) + np.absolute(sby)
        output = cv2.convertScaleAbs(grad_mag)
        
    return output

def toy_filtering_example():
    rows, cols = 4, 5
    image = np.reshape(np.arange(rows*cols), (rows,cols))
    print(image, image.shape)
    
    patches = sliding_window_view(image, window_shape=(3,3))
    #print(patches, patches.shape)
    
    kernel = np.array([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]])
    
    output = np.tensordot(patches, kernel, axes=[[2,3],[0,1]])
    print(output, output.shape)
    
    
    

###############################################################################
# MAIN
###############################################################################

def main(): 
    toy_filtering_example()
    exit()
    
    
    
    
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, 
                           kernel_size=3, bias=False,
                           padding="same")
    model = nn.Sequential(conv_layer)
    print(model)
        
    device = "cuda" # mps # cpu
    model = model.to(device)
    
    loss_fn = nn.MSELoss() # L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)        
    ])
    
    print("Filtering Options:")
    for index, item in enumerate(list(FilterType)):
        print(index, "-", item.value)
    chosen_index = int(input("Enter choice: "))
    filter_type = list(FilterType)[chosen_index]
    filter_size = int(input("Enter size: "))
           
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print("Random Torch Numbers:")
    print(b)
    print("Do you have Torch CUDA/ROCm?:", torch.cuda.is_available())
    print("Do you have Torch MPS?:", torch.mps.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("timm:", timm.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening the webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
                
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open the camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = do_filter(grayscale, filter_size, filter_type)
            
            gray_channel = np.expand_dims(grayscale, axis=-1)            
            data_input = data_transform(gray_channel)
            data_input = torch.unsqueeze(data_input, axis=0)
            
            sbx = cv2.Sobel(grayscale, 
                        ddepth=cv2.CV_64F, 
                        dx=1, dy=0, 
                        ksize=filter_size, 
                        scale=0.25)                        
            output_channel = np.expand_dims(sbx, axis=-1) 
            desired_output = data_transform(output_channel)
            desired_output = torch.unsqueeze(desired_output, axis=0)
            
            model.train()
            data_input = data_input.to(device)
            desired_output = desired_output.to(device)
            pred_output = model(data_input)
            loss = loss_fn(pred_output, desired_output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            out_image = pred_output.detach().cpu()
            out_image = out_image.numpy()
            out_image = out_image[0]
            out_image = np.transpose(out_image, [1,2,0])
            
            out_image = out_image*0.5 + 1.0
            
            cv2.imshow("PREDICTED", out_image)
                        
            print("Weights:", conv_layer.weight.detach().cpu().numpy())
               
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("OUTPUT", output)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'): 
                filter_size += 2
                print("Filter size:", filter_size)
            if key == ord('z'): 
                filter_size -= 2
                print("Filter size:", filter_size)
            filter_size = max(filter_size, 3)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) 
        
        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        
        key = -1
        while key == -1:
            cv2.imshow(windowTitle, image)

            # Wait for a keystroke to close the window
            key = cv2.waitKey(30)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    
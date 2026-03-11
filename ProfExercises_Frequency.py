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

left_mouse_down = False
right_mouse_down = False

def on_mouse(event, x, y, flags, param):
    global left_mouse_down
    global right_mouse_down
    
    def draw_to_mask_image(param, x, y, fill_value=0.0):
        mask_image = param[0]
        circle_radius = param[1]
        cv2.circle(mask_image, (x,y), circle_radius, fill_value, -1)
    
    if event == cv2.EVENT_LBUTTONUP:
        left_mouse_down = False
    elif event == cv2.EVENT_RBUTTONUP:
        right_mouse_down = False
    elif (event == cv2.EVENT_LBUTTONDOWN or 
          (left_mouse_down and event == cv2.EVENT_MOUSEMOVE)):
        left_mouse_down = True
        draw_to_mask_image(param, x, y, fill_value=0.0)
    elif (event == cv2.EVENT_RBUTTONDOWN or 
          (right_mouse_down and event == cv2.EVENT_MOUSEMOVE)):
        right_mouse_down = True
        draw_to_mask_image(param, x, y, fill_value=1.0)   

def to_numpy_complex(complex_image):
    return complex_image[:,:,0] + 1j*complex_image[:,:,1]

def to_complex_image(complex_data):
    return np.stack([
        np.real(complex_data), np.imag(complex_data)
    ], axis=2)

def complex_to_polar(complex_data):
    mag = np.abs(complex_data)
    phase = np.angle(complex_data)
    return mag, phase

def make_simple_complex(length=600):
    hl = length/2
    complex_data = np.zeros((length,length), dtype="complex")
    values = np.arange(-hl, hl, 1)
    complex_data[:] = values
    complex_data += np.reshape(1j*values, (-1, 1))    
    return complex_data

def compute_fourier(image, nonzeroRows=0):
    fimage = image.astype("float64")
    return cv2.dft(fimage, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=nonzeroRows)

def compute_inverse_fourier(complex_image, nonzeroRows=0):
    return cv2.idft(complex_image, 
                    flags=cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE,
                    nonzeroRows=nonzeroRows)
    
def make_display_magnitude(mag):
    display_mag = cv2.log(mag + 1.0)
    cv2.normalize(display_mag, display_mag, norm_type=cv2.NORM_MINMAX)
    return display_mag

def image_space_shift(image):
    rows, cols = image.shape[:2]
    row_powers = (-1)**np.arange(rows)[:,None]
    col_powers = (-1)**np.arange(cols)[None,:]
    neg_powers = row_powers * col_powers
    output = image * neg_powers
    return output

def polar_to_complex(mag, phase):
    return mag * np.exp(1j*phase)

def filter_with_fourier(image, kernel, mask_image=None):
    padded_rows = cv2.getOptimalDFTSize(image.shape[0]+kernel.shape[0]-1)
    padded_cols = cv2.getOptimalDFTSize(image.shape[1]+kernel.shape[1]-1)
    
    fimage = image.astype("float64")
    fkernel = kernel.astype("float64")
    
    padded_image = cv2.copyMakeBorder(fimage,
                                      0, padded_rows - fimage.shape[0],
                                      0, padded_cols - fimage.shape[1],
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    padded_kernel = cv2.copyMakeBorder(fkernel,
                                      0, padded_rows - fkernel.shape[0],
                                      0, padded_cols - fkernel.shape[1],
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    
    padded_image = image_space_shift(padded_image)
    padded_kernel = image_space_shift(padded_kernel)
    
    F = compute_fourier(padded_image)
    H = compute_fourier(padded_kernel)
    
    if mask_image is None:
        mask_image = np.ones(H.shape[:2], dtype="float64")
    
    cv2.namedWindow("H magnitude")
    mask_pack = [mask_image, 50]
    cv2.setMouseCallback("H magnitude", on_mouse, mask_pack)
    
    H_data = to_numpy_complex(H)
    H_mag, H_phase = complex_to_polar(H_data)
    #print(H_mag.shape, mask_image.shape)
    H_mag = H_mag * mask_image
    H_data = polar_to_complex(H_mag, H_phase)
    H = to_complex_image(H_data)
        
    G = cv2.mulSpectrums(F,H,0,conjB=False)
    
    def really_make_display_magnitude(fourier):
        return make_display_magnitude(complex_to_polar(to_numpy_complex(fourier))[0])
    
    display_F_mag = really_make_display_magnitude(F)
    display_H_mag = really_make_display_magnitude(H)
    display_G_mag = really_make_display_magnitude(G)
    
    cv2.imshow("F magnitude", display_F_mag)
    cv2.imshow("H magnitude", display_H_mag)
    cv2.imshow("G magnitude", display_G_mag)
    
    padded_output = compute_inverse_fourier(G, nonzeroRows=padded_image.shape[0])
    padded_output = image_space_shift(padded_output)
    
    sr = int(kernel.shape[0]/2)
    er = sr + image.shape[0]
    sc = int(kernel.shape[1]/2)
    ec = sc + image.shape[1]
    
    output = padded_output[sr:er, sc:ec]
    output /= 255.0
    
    return output, mask_image

def make_gaussian_filter(sidelen):
    gaussianCol = cv2.getGaussianKernel(ksize=sidelen, sigma=0)
    gaussianRow = np.transpose(gaussianCol)
    kernel = np.matmul(gaussianCol, gaussianRow)
    return kernel

def do_frequency(image, mask_image):
    kernel = make_gaussian_filter(3)
    output, mask_image = filter_with_fourier(image, kernel, mask_image)
    return output, mask_image

###############################################################################
# MAIN
###############################################################################

def main():  
    
    '''
    complex_data = make_simple_complex(600)
    mag, phase = complex_to_polar(complex_data)
    complex_image = to_complex_image(complex_data)
    
    cv2.normalize(complex_image, complex_image, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(mag, mag, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(phase, phase, norm_type=cv2.NORM_MINMAX)
    
    cv2.imshow("Original X", complex_image[...,0])
    cv2.imshow("Original Y", complex_image[...,1])
    cv2.imshow("Magnitude", mag)
    cv2.imshow("Phase", phase)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()    
    
    exit()
    '''
    
    mask_image = None
          
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
        ESC_KEY = 27
        key = -1
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #if mask_image is None:
            #    mask_image = np.ones(grayscale.shape, dtype="float64")
            
            output, mask_image = do_frequency(grayscale, mask_image)
            
            # Show the image
            cv2.imshow(windowName, image)
            cv2.imshow("OUTPUT", output)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('c'):
                mask_image[:,:] = 1.0

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
    
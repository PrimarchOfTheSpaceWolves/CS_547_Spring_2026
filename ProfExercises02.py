import cv2
import numpy as np

def main():
    image = np.zeros((480, 640, 3), dtype="uint8") # np.uint8
    
    image[:100, 150:, :] = 255
    image[200:300, :250] = (130, 50, 5) # BGR
    subimage = np.copy(image[:300, :300, :])
    
    image[:100, :100] = (0,0,255)
    
    fimage = image.astype("float64")
    
    uimage = cv2.convertScaleAbs(fimage) #, alpha=2.0, beta=1.0)
    #uimage = np.clip(np.round(fimage), 0, 255).astype("uint8")
    
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    
    gray_channel = np.expand_dims(gray, axis=-1)
    print(gray_channel.shape)
    batch_gray_channel = np.expand_dims(gray_channel, axis=0)
    print(batch_gray_channel.shape)
    gray_only_channel = np.squeeze(batch_gray_channel, axis=0)
    print(gray_only_channel.shape)
    
    #print(uimage.shape)
    #print(uimage.dtype)
    
    myimage = cv2.imread("images/test.png", cv2.IMREAD_COLOR)
    
    myimage = np.where(myimage <= 100, 100, myimage)
        
    #cv2.imshow("IMAGE", image)
    #cv2.imshow("SUBIMAGE", subimage)
    #cv2.imshow("FLOAT IMAGE", fimage)
    #cv2.imshow("MY IMAGE", myimage)
    #cv2.waitKey(-1)
    
    videocap = cv2.VideoCapture("images/noice.mp4")
    
    if not videocap.isOpened():
        print("HELP!")
        exit(1)
    
    key = -1
    while key == -1:
        _, frame = videocap.read()
        
        frame_cnt = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(videocap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_cnt == frame_index:
            videocap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cv2.imshow("NOICE", frame)
        key = cv2.waitKey(500) #30)
    videocap.release()

if __name__ == "__main__":
    main()
    
    
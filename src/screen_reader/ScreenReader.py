import numpy as np
from PIL import ImageGrab
import cv2
import time
from mss import mss
from PIL import Image

# -----
def screen_record_basic():
    while(True):
        printscreen_pil =  ImageGrab.grab()
        printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8') \
            .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
        # printscreen_numpy =   np.array(printscreen_pil,dtype='uint8') \
        #     .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
        cv2.imshow('window',printscreen_numpy)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
# -----


# -----
def screen_record():
    last_time = time.time()
    while (True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
# -----


# -----
mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}

sct = mss()

def screen_record_plus():
    while 1:
        sct.get_pixels(mon)
        img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        cv2.imshow('test', np.array(img))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
# -----

screen_record_basic()
# screen_record()
# screen_record_plus() #Giving an error currently.

import time
import webbrowser

import cv2
import numpy as np
from PIL import Image
from PIL import ImageGrab
from mss import mss
from pyautogui import press, typewrite, hotkey, click


def open_chrome():
    url = 'chrome://dino'
    # Mac
    # chrome_path = 'open -a /Application/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open_new(url)
    hotkey('ctrl', 'l')
    typewrite(url, interval=0.01)
    press('enter')
    time.sleep(0.2)
    click(960, 960)
    press('space')
    time.sleep(3)


def screen_record_basic():
    screen_pil = ImageGrab.grab()
    screen_numpy = np.array(screen_pil)

    return screen_numpy


def screen_record():
    last_time = time.time()
    while (True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()

        # cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# -----


# -----
mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}


def screen_record_plus():
    sct = mss()
    while 1:
        sct.grab(mon)
        clsIm = sct.cls_image
        img = Image.frombytes('RGB', (clsIm.width, clsIm.height), sct.cls_image)
        return np.array(img)

# -----

# open_chrome_2()
# screen_record_basic()
# screen_record()
# screen_record_plus() #Giving an error currently.

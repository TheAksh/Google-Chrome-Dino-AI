import numpy as np
from PIL import ImageGrab
import cv2
import time
from mss import mss
from PIL import Image
import webbrowser
import os
import subprocess
from pyautogui import press, typewrite, hotkey, keyDown, keyUp


def open_chrome():
    url = 'chrome://dino/'
    # Mac
    # chrome_path = 'open -a /Application/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open_new(url)


def open_chrome_2():
    # CHROME = r'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
    os.system('taskkill /im chrome.exe /F')
    # keyDown('win')
    # press('D')
    # keyUp('win')
    print("Sdes")
    os.system('start chrome "chrome://dino/"')
    # os.system("")
    x = 0
    while (x in range(10)):
        press('space')
        x += 1


# -----
def screen_record_basic():
    while (True):
        printscreen_pil = ImageGrab.grab()
        printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8') \
            .reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))
        # printscreen_numpy =   np.array(printscreen_pil,dtype='uint8') \
        #     .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))

        # cv2.imshow('window', printscreen_numpy)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        return cv2.imencode('window', printscreen_numpy)


# -----


# -----
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

sct = mss()


def screen_record_plus():
    while 1:
        sct.get_pixels(mon)
        img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)

        # cv2.imshow('test', np.array(img))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# -----

# open_chrome_2()
# screen_record_basic()
# screen_record()
# screen_record_plus() #Giving an error currently.

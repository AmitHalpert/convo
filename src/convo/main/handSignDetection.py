from src.convo.utils.handTrackingModule import HandDetector
from src.convo.utils.classificationModule import Classifier
from pynput.keyboard import Key, Controller
import sys
import cv2
import time
import numpy as np
import math

keyboard = Controller()


def start():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("../model/keras_model2.h5", "../model/labels2.txt")

    index = 0
    offset = 20
    imgSize = 300

    LastChar = ""
    writeDelay = 0
    folder = "../data/C"
    counter = 0

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "O", "P", "Q", "R", "W", "ENTER"]

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            writeDelay += time.thread_time()

        # keyboard control
        if writeDelay >= 700 and LastChar != labels[index] and labels[index] != "ENTER":
            label = labels[index]
            keyboard.type(label.lower())
            LastChar = labels[index]
            writeDelay = 0
        elif labels[index] == "ENTER" and writeDelay >= 700:
            keyboard.press(Key.enter)
            writeDelay = 0

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)
        if key == ord("q"):
            sys.exit("Q has been pressed. QUITING PROGRAM")

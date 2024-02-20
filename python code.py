import cv2
import numpy as np
import mediapipe as mp
import math
import tensorflow

class Detect_hands:
    
    def __init__(self, maxHands, mode=False):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = 0.5
        self.minTrackCon = 0.5

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        

    def predHands(self, img, draw=True, flipType=True):
       
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lmk in enumerate(handLms.landmark):
                    mylmList.append([int(lmk.x * w), int(lmk.y * h), int(lmk.z * w)])
                    xList.append(int(lmk.x * w))
                    yList.append(int(lmk.y * h))

                pass
                ## bbox
                min_x, max_x = min(xList), max(xList)
                min_y, max_y = min(yList), max(yList)
                boxW, boxH = max_x - min_x, max_y - min_y
                bbox = min_x, min_y, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                pass
                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 18, bbox[1] - 18),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 28, bbox[1] - 28), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands


class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        # setting the variables and loading the keras model
        self.model_path = modelPath
        
        np.set_printoptions(suppress=True)
        
        self.model = tensorflow.keras.models.load_model(self.model_path)

        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath

        pass
        if self.labels_path:
            alphabet_labels = open(self.labels_path, "r")
            self.list_labels = []
            for line in alphabet_labels:
                
                self.list_labels.append(line.strip())
            alphabet_labels.close()
       
        else:
            print("There are No Labels Found")

    def predict_alphabet(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        
        img_array = np.asarray(cv2.resize(img, (224, 224)))
        # Normalizing the image
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

        # Load the image
        self.data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(self.data)
        index_Val = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[index_Val]),
                        pos, cv2.FONT_HERSHEY_PLAIN, scale, color, 2)

        return list(prediction[0]), index_Val


# Main code

cap = cv2.VideoCapture(0)            #capturing the live video
detector = Detect_hands(maxHands=1)
model_name = "Model/keras_model.h5"
model_labels = "Model/labels.txt"
classifier = Classifier(model_name,model_labels)

offset = 20
img_Size = 300

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Space","Del"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.predHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        white_img = np.ones((img_Size, img_Size, 3), np.uint8) * 255
        crop_img = img[y - offset:y + h + offset, x - offset:x + w + offset]


        aspectRatio = h / w

        if aspectRatio > 1:
            # for the case where the height of the hand is bigger than the width
            k = img_Size / h
            wCal = math.ceil(k * w)
            img_Resize = cv2.resize(crop_img, (wCal, img_Size))
            img_Resize_Shape = img_Resize.shape
            pass
            wGap = math.ceil((img_Size - wCal) / 2)
            white_img[:, wGap:wCal + wGap] = img_Resize
            prediction, index = classifier.predict_alphabet(white_img, draw=False)
            print(prediction,index)
            

        else:
            # for the case where the width of the hand is bigger than the height
            k = img_Size / w
            hCal = math.ceil(k * h)
            img_Resize = cv2.resize(crop_img, (img_Size, hCal))
            img_Resize_Shape = img_Resize.shape
            pass
            hGap = math.ceil((img_Size - hCal) / 2)
            white_img[hGap:hCal + hGap, :] = img_Resize
            prediction, index = classifier.predict_alphabet(white_img, draw=False)
            print(prediction,index)
            

        cv2.rectangle(imgOutput, (x - offset, y - 70), (x +70, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -27), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageWhite", white_img)
        pass

    try:
        cv2.imshow('Image', imgOutput)
        cv2.waitKey(1)
        
    except cv2.error as e:
        print("error is:",e)


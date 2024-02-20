# American-Sign-Language-Detection

# Overview
The project is a hand tracking application that uses computer vision techniques to detect and recognize sign language gestures. The user makes hand gestures in front of a camera, and the application processes the video stream to identify the hand movements and recognize the corresponding sign language gesture. The output of the application is the recognized gesture displayed on the video stream.

# Approach
● The code uses Mediapipe library, to detect hands and landmarks in a video stream or image, and the TensorFlow library to classify the American Sign Language alphabet. The program initializes a video capture object and sets up the hand detector and the classifier. It then loops through the video frames and passes each frame to the hand detector to detect the hand landmarks. If a hand is detected, the landmarks are passed to the classifier to predict the alphabet sign. Finally, the predicted alphabet is displayed on the video stream along with the hand detection and landmarks.

● The Detect_hands class is used to detect hands and landmarks. It takes in two arguments, maxHands and mode, and initializes the Hands and DrawingUtils classes from the Mediapipe library. The maxHands argument specifies the maximum number of hands to detect, and the mode argument specifies whether to run in static image mode or video mode. The predHands method takes in an image frame, processes it using the Hands class, and returns the detected hands and landmarks.

● The Classifier class is used to classify the American Sign Language alphabet. It takes in the path to the model file and the path to the labels file as arguments. The predict_alphabet method takes in an image frame, processes it using the model, and returns the predicted alphabet.

# Result
My results demonstrate the effectiveness of our model in recognizing the ASL alphabet with high accuracy. However, further improvements can be made to reduce the confusion between similar signs.

Below are the images taken after testing my model in real time.
![image](https://github.com/vishalreddygarlapati/American-Sign-Language-Detection/assets/113934795/ca7482ef-6f26-4605-a43d-67c06df69ca0)
![image](https://github.com/vishalreddygarlapati/American-Sign-Language-Detection/assets/113934795/1526fc15-9eb3-404d-bce0-364f1e0cb56c)


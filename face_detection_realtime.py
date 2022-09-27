import cv2

cascade_classifier = cv2.CascadeClassifier('./static_face_detection/haarcascade_frontalface_alt.xml')

# real-time video (camera) - 0 means open default camera
video_capture = cv2.VideoCapture(0)

# size of video window
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    # returns the next video frame (the img is the important)
    ret, img = video_capture.read()

    # transform into grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    for (x, y, width, height) in detected_faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 10)

    cv2.imshow('Real-Time Face detection', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

video_capture.release()
cv2.releaseAllWindows()

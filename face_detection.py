import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('./static_face_detection/haarcascade_frontalface_alt.xml')

image = cv2.imread('./static_face_detection/image1.jpg')

# covert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, width, height) in detected_faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 10)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

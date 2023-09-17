import cv2 

img_file = 'car.jpg'
video = cv2.VideoCapture('tesla.mp4')

#arabaları okuma verileri yükleme
classifier_file='car_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:
     (read_successful, frame) = video.read()

     if read_successful:
        grayscaled_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     else:
        break 
     
     cars= car_tracker.detectMultiScale(grayscaled_frame)
     
     for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

     cv2.imshow('Car Detector',frame)

     cv2.waitKey(1)



"""
#Yüz algılamak için bir görüntü seçin
img = cv2.imread(img_file)

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier_file)

cars= car_tracker.detectMultiScale(black_n_white)
for (x,y,w,h) in cars:
   cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)



cv2.imshow('Car Detector',img)

cv2.waitKey()

print("Code Completed")
"""
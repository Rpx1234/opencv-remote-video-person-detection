import imagiz
import cv2
import numpy as np


server=imagiz.Server()
while True:
    message=server.receive()
    frame=cv2.imdecode(message.image,1)
    #cv2.imshow("",frame)

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()


    #resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (x1, y1, x2, y2) in boxes:
         # display the detected boxes in the color picture
         cv2.rectangle(frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)

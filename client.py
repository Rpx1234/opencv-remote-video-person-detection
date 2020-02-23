import imagiz
import cv2

#https://pypi.org/project/imagiz/
client=imagiz.Client("cc1",server_ip="putserverip")
vid=cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    r,frame=vid.read()
    if r:
        r, image = cv2.imencode('.jpg', frame, encode_param)
        client.send(image)
    else:
        break

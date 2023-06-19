import cv2
import picamera

camera = 0
video_path = "../test_video/cutScene_resize150.mp4"
# capture = cv2.VideoCapture(camera)
capture = cv2.VideoCapture(video_path)

while True:
    retval, frame = capture.read()
    
    if not retval:
        break
    
    resize_frame = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_CUBIC)
    
    cv2.imshow("resize frame", resize_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindow()
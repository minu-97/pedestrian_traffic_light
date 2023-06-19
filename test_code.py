import cv2
import numpy as np
import time
import re
from tflite_runtime.interpreter import Interpreter
from picamera.array import PiRGBArray
from picamera import PiCamera

min_confidence = 0.5
margin = 30

#file_name = "../test_image/dataset_signal_0.png"
label_name = "coco_labels.txt"
model_name = "detect.tflite"
number_light = 0

# initialize the camera and grab a reference to the raw camera capture
frame_width = 300
frame_height = 300
frame_resolution = [frame_width, frame_height]
frame_rate = 16
margin = 30
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))

# hsv signal filtering
lower_green = (30, 80, 20)
upper_green = (60, 255, 255)
lower_red = (170, 50, 70)
upper_red = (200, 255, 255)

# allow the camera to warmup
time.sleep(0.1)

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  #print(labels)
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

# Load tflite
labels = load_labels(label_name)
interpreter = Interpreter(model_name)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, 
    image = frame.array
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    
    # Loading image
    start_time = time.time()
    height, width, channels = image.shape
    # image = cv2.resize(img, (300, 300)) first input will be 300 300
    
    # Detecting objects
    outs = detect_objects(interpreter, image, min_confidence)

    font = cv2.FONT_HERSHEY_PLAIN
    color = (255, 0, 0)
    for out in outs:
        if out['class_id'] == 9 and out['score'] > min_confidence:
            number_light += 1
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = out['bounding_box']
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            label = '{:,.2%}'.format(out['score'])
            print(number_light, label)
            print(xmin, ymin, xmax, ymax)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, label, (xmin, ymin - 10), font, 1, color, 2)
            
            if number_light % 10 == 0:
                detected = image[xmin-10:xmax+10, ymin-10:ymax+10].copy()
                detected_hsv = cv2.cvtColor(detected, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(detected_hsv, lower_green, upper_green)
                mask_red = cv2.inRange(detected_hsv, lower_red, upper_red)
                cv2.imshow('detected green', mask_green)
                cv2.imshow('detected red', mask_red)
            
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    cv2.imshow('capture', image)

    end_time = time.time()
    process_time = end_time - start_time
    print("- A frame took {:.3f} seconds".format(process_time))


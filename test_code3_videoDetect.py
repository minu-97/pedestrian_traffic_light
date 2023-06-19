import cv2
import numpy as np
import time
import re
from threading import Thread
from tflite_runtime.interpreter import Interpreter
from threading import Thread
from playsound import playsound

min_confidence = 0.4
margin = 30
camera = 0

video_path = "../test_video/cutScene_resize150_300x10.mp4"
capture = cv2.VideoCapture(video_path)
label_name = "coco_labels.txt"
model_name = "detect.tflite"
index = 0

font = cv2.FONT_HERSHEY_PLAIN

signal_status = 0

# hsv signal filtering
lower_green = (60, 75, 80)
upper_green = (100, 255, 255)
lower_red = (0, 80, 80)
upper_red = (10, 255, 255)

# allow the camera to warmup
time.sleep(0.1)

def playbeep():
    while(True):
        if signal_status >= 7:
            playsound('beep_cut2.mp3')
        
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
#   print(labels)
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

# Loading image
start_time = time.time()
proc = Thread(target=playbeep, args=())
# proc.start()

while True:
    retval, frame = capture.read()
    
    if not retval:
        break
    
    resize_frame = cv2.resize(frame, (300, 300), interpolation = cv2.INTER_CUBIC)
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    
    # Loading image
    start_time = time.time()
    height, width, channels = resize_frame.shape
    # image = cv2.resize(img, (300, 300)) first input will be 300 300
    
    # Detecting objects
    outs = detect_objects(interpreter, resize_frame, min_confidence)

    font = cv2.FONT_HERSHEY_PLAIN
    
    for out in outs:
        if out['class_id'] == 9 and out['score'] > min_confidence:
            index += 1
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = out['bounding_box']
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            label = '{:,.2%}'.format(out['score'])
#             print(index, label)
            
            if signal_status >= 7 :
                color = (255, 255, 255)
            else:
                color = (255, 0, 0)
                
            cv2.rectangle(resize_frame, (xmin-15, ymin-30), (xmax+15, ymax+30), color, 2)
#             cv2.putText(resize_frame, label, (xmin, ymin - 15), font, 1, color, 2)
            
            text = "detected"
            cv2.putText(resize_frame, text, (margin, margin), font, 2, color, 2)
        
            end_time = time.time()
            process_time = end_time - start_time
#             print("=== A frame took {:.3f} seconds".format(process_time))
             

#             if index % 4 == 0:
            detected = resize_frame[ymin:ymax, xmin:xmax].copy()
            detected_hsv = cv2.cvtColor(detected, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(detected_hsv, lower_green, upper_green)
            mask_red = cv2.inRange(detected_hsv, lower_red, upper_red)
#             cv2.imshow('detected green', mask_green)
#             cv2.imshow('detected red', mask_red)
            
            mask_rh = len(mask_red)
            mask_rw = len(mask_red[0])
            detected_count_r = sum(sum(mask_red/255))
            mask_area_r = mask_rh * mask_rw
            proportion_r = detected_count_r / mask_area_r * 100
            
            mask_gh = len(mask_green)
            mask_gw = len(mask_green[0])
            detected_count_g = sum(sum(mask_green/255))
            mask_area_g = mask_gh * mask_gw
            proportion_g = detected_count_g / mask_area_g * 100
            
            if proportion_r > 2 :
                signal_status -= 5
                if signal_status < 0 :
                    signal_status = 0
                print("[R E D] STACK -5", signal_status)
            elif proportion_g > 10 :
                signal_status += 1
                if signal_status > 10 :
                    signal_status = 10
                print("[GREEN] STACK +1", signal_status)
            else:
                print("can't determine", signal_status)
            
            if signal_status >= 7 :
                print("Green Light!")
            else:
                print("Red Light!")


#                 print("proportion RED : ", round(proportion_r, 2))
#                 print("proportion GREEN : ", round(proportion_g, 2))
                
                 
#             cv2.imshow('detected area', detected)

#             test code
#             cvt_hsv = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2HSV)
#             mask_green2 = cv2.inRange(cvt_hsv, lower_green, upper_green)
#             mask_red2 = cv2.inRange(cvt_hsv, lower_red, upper_red)
#             cv2.imshow('HSV domain green', mask_green2)
#             cv2.imshow('HSV domain red', mask_red2)                

    cv2.imshow("CAMERA", resize_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
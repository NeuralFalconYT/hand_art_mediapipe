# this code not optamized just for fun
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import time
import shutil
import random
model_path = './hand_landmarker.task'
# video_path= "./1.mp4"
video_path="0"
# fixed_width,fixed_height = 640,480
# fixed_width,fixed_height = 1280,720
fixed_width,fixed_height =  1920,1080
# fixed_width,fixed_height =  1080,1920
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2 )
landmarker= HandLandmarker.create_from_options(options)


#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
  annotated_image=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  return annotated_image
def adjust_gap_between_points(points, threshold=1.2):
    gap=50
    finger_points = {
        1: [1, 2, 3, 4],  # Thumb
        2: [5, 6, 7, 8],  # Index finger
        3: [9, 10, 11, 12],  # Middle finger
        4: [13, 14, 15, 16],  # Ring finger
        5: [17, 18, 19, 20]   # Little finger
    }
    store=[]
    # Loop through each hand's landmarks
    for hand_landmarks in points:
        # Loop through each landmark in the dictionary
        temp={}
        for idx, (x, y, z) in hand_landmarks.items():
            if idx not in [1,5,9,13,17]:
                x+=gap
                y-=gap
            temp[idx]=(x,y,z)
            # print(temp)
        store.append(temp)
    # print(store)
    # return store
    return points


def find_points(detection_result, image=None):
    hand_landmarks_list = detection_result.hand_landmarks
    store = []
    
    if image is not None:
        height, width, _ = image.shape  # Get the actual image dimensions
    
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            temp = {}
            
            for index,landmark in enumerate(hand_landmarks):
                # Convert normalized coordinates to pixel coordinates
                x_pixel = int(landmark.x * width) # Convert x to pixel value
                y_pixel = int(landmark.y * height) # Convert y to pixel value
                z_value = landmark.z * 100 # z Example scaling for depth, or use as-is
                temp[index] = (x_pixel, y_pixel, z_value)      
            store.append(temp)
        
        # print(store)
        adjusted_store = adjust_gap_between_points(store, threshold=1.5)
        return adjusted_store
    else:
        print("Image is None, cannot calculate real coordinates.")
        return None

# def own_draw(points, image):
#     color = (255,255,255)  # White color for the circle
#     radius = 5  # Small radius for the circle
#     thickness = 2  # Outline thickness (positive value for non-filled circle)
    
#     # 'points' is a list of dictionaries, each dictionary containing 20 landmarks
#     # for one hand: {1: (x, y, z), ..., 20: (x, y, z)}
    
#     # Loop through each hand's landmarks
#     for hand_landmarks in points:
#         # Loop through each landmark in the dictionary
#         for idx, (x, y, z) in hand_landmarks.items():
#             # Draw a circle at the (x, y) coordinate
#             cv2.circle(image, (int(x), int(y)), radius, color, thickness)
    
#     # Optional: Apply a slight Gaussian blur for smoothing, if circles look pixelated
#     image = cv2.GaussianBlur(image, (3, 3), 0)  # Small blur to smooth edges
    
#     return image

from PIL import Image, ImageDraw


def draw_small_circles(pillow_image, points):
    # Convert the OpenCV image (NumPy array) to a Pillow image
    draw = ImageDraw.Draw(pillow_image)

    color = (255, 255, 255)  # White color for the circle
    radius = 4  # Small radius for the circle
    # points format is like
    # points = [{1: (x1, y1, z1), ..., 20: (x20, y20, z20)}, {1: (x1, y1, z1), ..., 20: (x20, y20, z20)}]
    # Loop through each hand's landmarks
    for hand_landmarks in points:
        # Loop through each landmark in the dictionary
        for idx, (x, y, z) in hand_landmarks.items():
            # Draw a circle at the (x, y) coordinate using a bounding box
            left_up_point = (int(x) - radius, int(y) - radius)
            right_down_point = (int(x) + radius, int(y) + radius)
            draw.ellipse([left_up_point, right_down_point], outline=color, width=1)
    return pillow_image


from PIL import Image, ImageDraw

def draw_curved_line(draw, p0, p1, control, steps=100, color=(255, 255, 255), width=2):
    """ Draw a quadratic Bezier curve from p0 to p1 with a control point. """
    for t in range(steps + 1):
        t /= steps  # Normalize t from 0 to 1
        # Calculate the Bezier curve point
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * control[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * control[1] + t ** 2 * p1[1]
        draw.line([(x, y), (x, y)], fill=color, width=width)

def curved_line(pillow_image, points):
    finger_points = {
        1: [1, 2, 3, 4],  # Thumb
        2: [5, 6, 7, 8],  # Index finger
        3: [9, 10, 11, 12],  # Middle finger
        4: [13, 14, 15, 16],  # Ring finger
        5: [17, 18, 19, 20]   # Little finger
    }

    draw = ImageDraw.Draw(pillow_image)
    color = (255, 255, 255)  # White color for the curve line
    curve_width = 4  # Width of the curve line

    # Loop through each hand's landmarks
    for hand_landmarks in points:
        # Loop through each finger
        for finger, landmarks in finger_points.items():
            # Ensure we have valid landmark indices
            if all(idx in hand_landmarks for idx in landmarks):
                # Draw curves between landmarks with alternating control points
                for i in range(len(landmarks) - 1):
                    p0 = hand_landmarks[landmarks[i]]
                    p1 = hand_landmarks[landmarks[i + 1]]

                    # Determine control point for the curve
                    if i % 2 == 0:  # Left curve for odd index
                        control = (p0[0] + 5, (p0[1] + p1[1]) / 2 - 2)  # Adjust values for left curve
                        control_1=(p0[0] - 5, (p0[1] + p1[1]) / 2 + 2)
                    else:  # Right curve for even index
                        control = (p0[0] - 5, (p0[1] + p1[1]) / 2 + 2)  # Adjust values for right curve
                        control_1=(p0[0] + 5, (p0[1] + p1[1]) / 2 - 2)
                    draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
                    #skip the last point
                    if i!=2:
                        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control_1, color=color, width=curve_width)

    return pillow_image
def only_curved_line(pillow_image,points):
    color = (255, 255, 255)  # White color for the curve line
    draw=ImageDraw.Draw(pillow_image)
    curve_width = 15 # Width of the curve line
    front=False
    back=False
    for hand_landmarks in points:
        x1,y1,_=hand_landmarks[4]
        x2,y2,_=hand_landmarks[5]
        print(x1,x2)
        print(y1,y2)
        if x2>x1:
            front=True
            back=False
        else:
            back=True
            front=False
            
            
        x1,y1,_=hand_landmarks[4]
        x2,y2,_=hand_landmarks[2]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] + 7, (p0[1] + p1[1]) / 2 - 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
        
        
        x1,y1,_=hand_landmarks[0]
        x2,y2,_=hand_landmarks[2]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] - 7, (p0[1] + p1[1]) / 2 + 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
        
        
        x1,y1,_=hand_landmarks[8]
        x2,y2,_=hand_landmarks[11]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] + 5, (p0[1] + p1[1]) / 2 - 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
        
        x1,y1,_=hand_landmarks[12]
        x2,y2,_=hand_landmarks[15]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] + 5, (p0[1] + p1[1]) / 2 - 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
        
        x1,y1,_=hand_landmarks[16]
        x2,y2,_=hand_landmarks[19]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] + 5, (p0[1] + p1[1]) / 2 - 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width)
        
        x1,y1,_=hand_landmarks[1]
        x2,y2,_=hand_landmarks[16]
        p0=(x1,y1)
        p1=(x2,y2)
        control = (p0[0] + 5, (p0[1] + p1[1]) / 2 - 2) 
        draw_curved_line(draw, (p0[0], p0[1]), (p1[0], p1[1]), control, color=color, width=curve_width+6)
        
        # radius = 10   
        # a_noise=random.randint(0, 10)
        # b_noise=random.randint(0, 10)
        # a_noise=0
        # b_noise=0
        # x1,y1,_=hand_landmarks[6]
        # center=(x1,y1)
        #     # Calculate the bounding box for the rectangle
        # top_left = (center[0] - radius-a_noise, center[1] - radius-a_noise)
        # bottom_right = (center[0] + radius+b_noise, center[1] + radius+b_noise)
        # # Draw the rectangle
        # draw.rectangle([top_left, bottom_right], outline=color, width=1)
    

        
        # x1,y1,_=hand_landmarks[7]
        # center=(x1,y1)
        #     # Calculate the bounding box for the rectangle
        # top_left = (center[0] - radius-a_noise, center[1] - radius-a_noise)
        # bottom_right = (center[0] + radius+b_noise, center[1] + radius+b_noise)
        # # Draw the rectangle
        # draw.rectangle([top_left, bottom_right], outline=color, width=1)
        
        #draw the rectangle
        x1,y1,_=hand_landmarks[17]
        x2,y2,_=hand_landmarks[19]
        x3=x1+(x2-x1)
        y3=y1+50
        x1,y1,_=hand_landmarks[7]
        p0=(x1,y1)
        p1=(x3,y3)
        pillow_image=draw_rectangle(pillow_image, p0, p1)
        
        
        #draw the rectangle
        x1,y1,_=hand_landmarks[1]
        x2,y2,_=hand_landmarks[2]
        p0=(x1-5,y1)
        p1=(x2,y2+10)
        pillow_image=draw_rectangle(pillow_image, p0, p1)
        
        #draw the rectangle
        x1,y1,_=hand_landmarks[2]
        x2,y2,_=hand_landmarks[3]
        p0=(x1-5,y1-10)
        p1=(x2,y2)
        pillow_image=draw_rectangle(pillow_image, p0, p1)
        
        
        # x1,y1,_=hand_landmarks[7]
        # x2,y2,_=hand_landmarks[8]
        # p0=(x1-5,y1)
        # p1=(x2,y2+10)
        # pillow_image=draw_rectangle(pillow_image, p0, p1)
        
        
        # x1,y1,_=hand_landmarks[6]
        # x2,y2,_=hand_landmarks[7]
        # p0=(x1-5,y1)
        # p1=(x2,y2+10)
        # pillow_image=draw_rectangle(pillow_image, p0, p1)
        #draw the circle
        x1,y1,_=hand_landmarks[3]
        x2,y2,_=hand_landmarks[6]
        #Find the center of the circle
        if front:
            center = (x1 + (x2 - x1) // 2, y1-60)  # Correct calculation for center
        else:
            print("back")
            center = (x1 + (x2 - x1) // 2, y1-60)
        radius = max(0,  abs((x2 - x1) // 2))  # Ensure radius is non-negative
        pillow_image=draw_circle(pillow_image, center, radius, color=(255, 255, 255), width=2)
        
        
         #draw the circle
        x1,y1,_=hand_landmarks[15]
        x2,y2,_=hand_landmarks[20]
        x3,y3,z3=hand_landmarks[14]
        if front:
            center = (x3+30,y3)
        else:
            center = (x3-30,y3) 

        radius = max(0,  abs((x2 - x1) // 2))  # Ensure radius is non-negative
        pillow_image=draw_circle(pillow_image, center, radius, color=(255, 255, 255), width=2)
        
        x1,y1,_=hand_landmarks[1]
        text='X'
        a_noise=random.randint(0, 0)
        b_noise=random.randint(0, 100)
        if front:
            # fontsize little biger and text bold
            draw.text((x1+30+a_noise,y1-b_noise), text, fill=color, font=None)
            draw.text((x1+20+a_noise,y1-b_noise), text, fill=color, font=None)
            draw.text((x1+40+a_noise,y1-20-b_noise), text, fill=color, font=None)   
        else:
            draw.text((x1-30-a_noise,y1-b_noise), text, fill=color, font=None)
            draw.text((x1-20-a_noise,y1-b_noise), text, fill=color, font=None)
            draw.text((x1-40-a_noise,y1-20-b_noise), text, fill=color, font=None)
    return pillow_image

def draw_rectangle(pillow_image, p0, p1, color=(255, 255, 255), width=1):
    draw = ImageDraw.Draw(pillow_image)
    
    # Calculate the coordinates for the rectangle
    top_left = (min(p0[0], p1[0]), min(p0[1], p1[1]))
    bottom_right = (max(p0[0], p1[0]), max(p0[1], p1[1]))
    
    # Draw the rectangle
    draw.rectangle([top_left, bottom_right], outline=color, width=width)

    return pillow_image

def draw_circle(pillow_image, center, radius, color=(255, 255, 255), width=2):
    draw = ImageDraw.Draw(pillow_image)
    
    # Calculate the bounding box for the circle
    top_left = (center[0] - radius, center[1] - radius)
    bottom_right = (center[0] + radius, center[1] + radius)
    
    # Draw the circle (outline only)
    draw.ellipse([top_left, bottom_right], outline=color, width=width)

    return pillow_image         
def own_draw(points, image):
    pillow_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    curve_finger=curved_line(pillow_image, points)
    curve_line_image=only_curved_line(curve_finger,points)
    circle_image=draw_small_circles(curve_line_image, points)
    image = cv2.cvtColor(np.array(circle_image), cv2.COLOR_RGB2BGR)
    return image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# Initialize VideoCapture object
# Initialize VideoCapture object
if video_path == "0":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fixed_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fixed_height)
else:
    cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # If fps is not properly detected, set a default value
    fps = 30.0

# Video save settings
output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_filename, fourcc, fps, (fixed_width, fixed_height))

start_time = time.time()
previous_timestamp_ms = 0

# Processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # If webcam input, flip the frame horizontally
    if video_path == "0":
        frame = cv2.flip(frame, 1)

    # Timestamp management
    elapsed_time = time.time() - start_time
    frame_timestamp_ms = int(elapsed_time * 1000)  # Convert to milliseconds

    # Ensure the timestamp is monotonically increasing
    if frame_timestamp_ms <= previous_timestamp_ms:
        frame_timestamp_ms = previous_timestamp_ms + 1
    previous_timestamp_ms = frame_timestamp_ms

    # Convert frame to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image and perform hand detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # Find hand landmark points and draw custom landmarks
    points = find_points(detection_result, rgb_image)
    draw_image = own_draw(points, frame)

    # Resize frame for saving and ensure correct size
    save_image = cv2.resize(draw_image, (fixed_width, fixed_height))

    # Write the processed frame to the output video file
    out.write(save_image)

    # Show the frame with custom drawings
    cv2.imshow('Hand Landmarks', draw_image)

    # Exit loop if 'Esc' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the VideoCapture and VideoWriter objects, and close all OpenCV windows
cap.release()
out.release()  # Ensure that the video writer is released properly
cv2.destroyAllWindows()





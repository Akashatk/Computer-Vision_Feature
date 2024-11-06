#pip install code: pip install opencv-python mediapipe numpy screen-brightness-control pycaw comtypes pillow

import cv2
import mediapipe as mp
import numpy as np
from screen_brightness_control import set_brightness
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from PIL import ImageGrab  # For full-screen capture

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up audio volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()


# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_dips = [
        mp_hands.HandLandmark.INDEX_FINGER_DIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp_hands.HandLandmark.RING_FINGER_DIP,
        mp_hands.HandLandmark.PINKY_DIP
    ]
    
    raised_fingers = 0
    for tip, dip in zip(finger_tips, finger_dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            raised_fingers += 1
    return raised_fingers

# Function to detect thumbs-up gesture
def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    return (thumb_tip.y < thumb_ip.y < thumb_mcp.y < index_mcp.y)
def is_thumbs_open(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    return (thumb_tip.x < thumb_ip.x < thumb_mcp.x < index_mcp.x)
# Function to capture full-screen screenshot
def capture_fullscreen_screenshot():
    screenshot = ImageGrab.grab()
    screenshot.save("fullscreen_screenshot.png")
    print("Full-screen screenshot saved as fullscreen_screenshot.png")
def capture_screenshot(frame):
    screenshot_name = 'screenshot.png'
    cv2.imwrite(screenshot_name, frame)
    print(f"Screenshot saved as {screenshot_name}")

# Open video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raised_fingers = count_raised_fingers(hand_landmarks)
                
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
                if is_thumbs_up(hand_landmarks):
                        print("Thumbs-up detected. Exiting program.")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                if raised_fingers == 1:
                    ss_flag=True
                    # Control Volume
                    volume_level = np.interp(distance, [20, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(volume_level, None)
                    vol_percentage = int(np.interp(volume_level, [min_vol, max_vol], [0, 100]))
                    cv2.putText(frame, f'Volume: {vol_percentage}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                elif raised_fingers == 2:
                    ss_flag=True
                    # Control Brightness
                    brightness_level = np.interp(distance, [20, 200], [0, 100])
                    set_brightness(int(brightness_level))
                    cv2.putText(frame, f'Brightness: {int(brightness_level)}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif raised_fingers == 3:
                    # Capture Full-Screen Screenshot
                    # Check for thumbs-up gesture to exit
                    if(ss_flag):
                        if is_thumbs_open(hand_landmarks):
                            capture_fullscreen_screenshot()
                            cv2.putText(frame, 'Full-Screen Screenshot Captured', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            capture_screenshot(frame)
                            cv2.putText(frame, 'Screenshot Captured', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        ss_flag=False
                elif raised_fingers == 4:
                    ss_flag=True
                    if is_thumbs_open(hand_landmarks):
                        cv2.putText(frame, 'Mic Umuted', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        # Mute Microphone (requires OS-specific implementation)
                        cv2.putText(frame, 'Mic Muted', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

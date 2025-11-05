import cv2
import time
from ultralytics import YOLO
import os

# For checking and downloading YOLO model if it is not in the system
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  
else:
    model = YOLO(MODEL_PATH)

# The green signal parameters
BASE_GREEN_TIME = 15   
MAX_GREEN_TIME = 40    
MIN_GREEN_TIME = 10    

def calculate_green_time(vehicle_count):
    # Changing the green signal timing according to the number of cars detected
    if vehicle_count < 5:
        return MIN_GREEN_TIME
    elif vehicle_count < 15:
        return BASE_GREEN_TIME
    else:
        return MAX_GREEN_TIME

def calculate_red_time(vehicle_count):
    # Changing the red signal timing according to the number of cars detected
    if vehicle_count < 5:
        return MAX_GREEN_TIME 
    elif vehicle_count < 15:
        return BASE_GREEN_TIME  
    else:
        return MIN_GREEN_TIME  
def process_video(video_source="sample_traffic.mp4"):

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    vehicle_classes = [2, 3, 5, 7]  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read the frame.")
            break

        # Runs object detection in the frame
        results = model(frame)[0]

        boxes = results.boxes
        vehicle_count = 0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for i in range(len(classes)):
                cls = classes[i]
                if cls in vehicle_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    conf = confidences[i]
                    label = f"{cls} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculates the timing of the green signal
        green_time = calculate_green_time(vehicle_count)

        # Displays vehicle count and green signal time on the frame
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Green Time: {green_time}s', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Traffic Monitoring', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # For red signal
            red_time = calculate_red_time(vehicle_count)
            print(f"\n🔴 Red signal ON for {red_time} seconds based on {vehicle_count} vehicles.")
            last_remaining = None
            start_time = time.time()
            while time.time() - start_time < red_time:
                remaining = max(0, int(red_time - (time.time() - start_time)))

                if remaining != last_remaining:
                    print(f"Countdown: {remaining}s left")
                    last_remaining = remaining

                # Prints the countdown of the red and green signals
                temp_frame = frame.copy()
                cv2.putText(temp_frame, f'Red Light: {remaining}s left', (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # Red
                cv2.putText(temp_frame, f'Green Time: {green_time}s', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green
                cv2.imshow('Traffic Monitoring', temp_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print("🟢 Signal switched to GREEN.")

        elif key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("sample_traffic.mp4")
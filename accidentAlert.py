from ultralytics import YOLO
import cv2
import time
import requests

# load model 
model = YOLO("best.pt")

# video open
cap = cv2.VideoCapture("videoplayback.mp4")

# report
report_file = open("accident_report.txt", "w")
frame_count = 0

# Telegram Bot settings
TELEGRAM_BOT_TOKEN = "7924575317:AAHdcVLqMIcaY53c5wm3eSSEchbeJ0xFb30"  # Token 
TELEGRAM_CHAT_ID = "7734615619"  # Chat ID  
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"  # sendMessage
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,  
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Send message to Telegram ")
        else:
            print(f"faill: {response.text}")
    except Exception as e:
        print(f"faill: {e}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    
    results = model(frame)

    
    probs = results[0].probs
    class_names = results[0].names

    #Accident
    accident_idx = None
    for i, name in enumerate(class_names.values()):
        if isinstance(name, str) and "accident" in name.lower():
            accident_idx = i
            break

    if accident_idx is None:
        print("ไม่พบ class 'Accident' ในโมเดล")
        break

    accident_prob = probs.data[accident_idx]

    #  "Accident" < 90% 
    if accident_prob > 0.9:
        text = f"Accident Detected ({accident_prob:.2f})"
        color = (0, 0, 255)

        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)

       
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        report_file.write(f"Frame {frame_count} at {timestamp}: Accident ({accident_prob:.2f})\n")

        # send message to Telegram
        message = f"🚨 ກວດພົບອຸບັດຕິເຫດ\n: {frame_count}\nເວລາ: {timestamp}\nຄວາມເປັນໄປໄດ້: {accident_prob:.2f}"
        send_telegram_message(message)

        
        cv2.imwrite(f"accident_frame_{frame_count}.jpg", frame)
    else:
        text = f"No Accident ({accident_prob:.2f})"
        color = (0, 255, 0)

    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # result 
    cv2.imshow("Car Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
report_file.close()
cv2.destroyAllWindows()
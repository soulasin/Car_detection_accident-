# Car Accident Detection and Telegram Alert System

This project develops an automated system to detect car accidents from video footage using a YOLOv8 model and sends real-time alerts (with images and mock locations) to authorities via Telegram. The system aims to reduce response time to accidents, enhancing road safety in society.

### Objectives
- Detect car accidents from video with ≥80% confidence using YOLOv8.
- Send an alert with an image and mock location to Telegram within 10 seconds of detection.
- Reduce emergency response time from 15-30 minutes to 5-10 minutes.

  ### Features
- Real-time accident classification using YOLOv8 (`best.pt` model).
- Draws a mock circle on detected accident frames.
- Saves detection reports and images locally.
- Sends a single Telegram alert with the first detected image and a mock location .

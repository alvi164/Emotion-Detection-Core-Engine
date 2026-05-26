import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting ULTRA-LIGHT mode...")

# Use a smaller frame size to speed up processing
while True:
    ret, frame = cap.read()
    if not ret: break

    # RESIZE: Shrink the image to 50% size for the AI (Massive speed boost)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    try:
        # We only look for Emotion and Gender (Age is the heaviest model, skipping it helps stability)
        results = DeepFace.analyze(small_frame, 
                                   actions=['emotion', 'gender'], 
                                   enforce_detection=False,
                                   detector_backend='opencv',
                                   silent=True)
        
        for res in results:
            # Upscale coordinates back to full size
            x, y, w, h = [v * 2 for v in [res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']]]
            emotion = res['dominant_emotion']
            gender = res['dominant_gender']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}, {emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    except:
        pass

    cv2.imshow('ROG Speed Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
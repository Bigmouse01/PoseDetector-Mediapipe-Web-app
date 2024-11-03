from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def is_posture_correct(nose, left_shoulder, right_shoulder):
    midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
    midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
    midpoint_z = (left_shoulder[2] + right_shoulder[2]) / 2
    
    # Check if the nose is aligned with the midpoint of the shoulders
    # You can define a threshold to allow minor deviations
    threshold = 0.9
    trs=0.57
    
      # Adjust as necessary (for z-coordinate)

    is_aligned = (
        abs(nose[0] - midpoint_x) < threshold and
        abs(nose[1] - midpoint_y) < threshold and
        abs(nose[2] - midpoint_z) < threshold and
        abs(nose[0]- left_shoulder[0])<threshold and
        abs(nose[1]- left_shoulder[1])<threshold and
        abs(nose[2]- left_shoulder[2])<threshold and
        abs(nose[0]- right_shoulder[0])<trs and
        abs(nose[1]-right_shoulder[1])<trs and
        abs(nose[2]- right_shoulder[2])<trs 
        
    )

    return is_aligned
    # Your existing posture check code...

def generate_frames():
    cap = cv2.VideoCapture(0)
    time1 = time.time()

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get landmarks
                nose = (
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z
                )
                left_shoulder = (
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                )
                right_shoulder = (
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                )
                
                left_elbow_visible = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility > 0.65
                right_elbow_visible = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > 0.65

                if left_elbow_visible or right_elbow_visible:
                    cv2.putText(frame, "Move closer to the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    posture_correct = is_posture_correct(nose, left_shoulder, right_shoulder)
                    if posture_correct:
                        cv2.putText(frame, "Posture: Correct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Posture: Incorrect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            time2 = time.time()
            fps = 1.0 / (time2 - time1)
            time1 = time2
            cv2.putText(frame, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

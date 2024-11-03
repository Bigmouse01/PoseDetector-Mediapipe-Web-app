import cv2
import mediapipe as mp
import time as time

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def is_posture_correct(nose, left_shoulder, right_shoulder):
    """
    Check if the person's sitting posture is correct based on the nose and shoulder positions.

    Args:
    nose (tuple): (x, y, z) coordinates of the nose.
    left_shoulder (tuple): (x, y, z) coordinates of the left shoulder.
    right_shoulder (tuple): (x, y, z) coordinates of the right shoulder.

    Returns:
    bool: True if the posture is correct, False otherwise.
    """
    # Calculate the midpoint of the shoulders
    midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
    midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
    midpoint_z = (left_shoulder[2] + right_shoulder[2]) / 2
    
    # Check if the nose is aligned with the midpoint of the shoulders
    # You can define a threshold to allow minor deviations
    threshold = 0.60
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

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    time1=time.time()

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            # Draw landmarks
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get the landmarks for nose, left shoulder, and right shoulder
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
                # Check if elbows are visible
                left_elbow_visible = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility > 0.55
                right_elbow_visible = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > 0.55

                if left_elbow_visible or right_elbow_visible:
                    cv2.putText(frame, "Move closer to the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:


                # Check posture
                    posture_correct = is_posture_correct(nose, left_shoulder, right_shoulder)

                # Display the result
                    if posture_correct:
                        cv2.putText(frame, "Posture: Correct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Posture: Incorrect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            time2=time.time()
            fps=1.0/(time2-time1)
            time1=time2
            #To diplay this fps window we dot his nigger
            cv2.putText(frame, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)           

            # Display the frame
            cv2.imshow('Posture Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #myvenv\Scripts\activate-Activates Virtual environment.
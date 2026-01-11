import cv2
import cv2.aruco as aruco
import numpy as np
import time


# --- AI CONTROLLER CLASS ---
class PIDController:
    def __init__(self, kp=0.1):
        self.kp = kp
        self.prev_error = 0

    def compute(self, error):
        p_term = self.kp * error
        d_term = 0.1 * (error - self.prev_error)
        self.prev_error = error
        return p_term + d_term


# --- UTILS ---
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return np.degrees(np.arctan2(delta_y, delta_x))


def draw_vector_arrow(frame, start_point, magnitude_x, magnitude_y):
    end_point = (
        int(start_point[0] + magnitude_x * 2),
        int(start_point[1] + magnitude_y * 2),
    )
    cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)


def draw_docking_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = (0, 255, 0)
    if int(time.time() * 4) % 2 == 0:
        color = (200, 255, 200)

    text = "CONNECTION ESTABLISHED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
    text_x = (w - text_size[0]) // 2

    cv2.putText(frame, text, (text_x, h // 2 + 10), font, 1.2, color, 3)
    cv2.putText(
        frame,
        "CHARGING SEQUENCE ACTIVE",
        (text_x + 50, h // 2 + 45),
        font,
        0.6,
        (255, 255, 255),
        1,
    )


def draw_sci_fi_hud(frame, level, confidence, start_time, status_msg, status_color):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # --- 1. SYSTEM STATUS (TOP LEFT) ---
    cv2.putText(
        frame, "SYSTEM STATUS:", (30, 50), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1
    )
    cv2.putText(
        frame, status_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
    )

    # --- 2. BATTERY BAR ---
    cv2.rectangle(frame, (20, h // 2 - 100), (40, h // 2 + 100), (50, 50, 50), -1)
    fill_h = int(200 * (level / 100))
    bat_color = (0, 255, int(2.55 * level))
    cv2.rectangle(frame, (22, h // 2 + 100 - fill_h), (38, h // 2 + 100), bat_color, -1)

    cv2.putText(
        frame, "PWR", (15, h // 2 + 120), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1
    )
    cv2.putText(
        frame,
        f"{int(level)}%",
        (12, h // 2 + 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        bat_color,
        1,
    )

    # --- 3. FLIGHT TIMER ---
    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    timer_text = f"T+ {minutes:02}:{seconds:02}"

    cv2.putText(
        frame,
        "FLIGHT TIME",
        (w - 180, 50),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame,
        timer_text,
        (w - 180, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    # --- 4. AI CONFIDENCE METER ---
    cv2.putText(
        frame,
        f"AI CONFIDENCE: {int(confidence)}%",
        (w - 300, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.line(frame, (w - 300, 110), (w - 50, 110), (100, 100, 100), 2)
    conf_color = (0, 255, 0) if confidence > 80 else (0, 165, 255)
    cv2.line(
        frame, (w - 300, 110), (w - 300 + int(2.5 * confidence), 110), conf_color, 2
    )

    # --- 5. CENTER SCANNER ---
    if status_msg == "SCANNING...":
        cv2.circle(frame, (center_x, center_y), 40, (0, 255, 255), 1)


def start_advanced_docking():
    cap = cv2.VideoCapture(0)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    battery_level = 100.0
    start_time = time.time()
    last_drain_time = time.time()
    pid_x = PIDController(kp=0.4)
    pid_y = PIDController(kp=0.4)

    print("System Online. Show Marker to Dock.")

    while True:
        ret, frame = cap.read()
        if not ret:de
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)

        # Default State
        status_msg = "SCANNING..."
        status_color = (0, 255, 255)  # Yellow
        confidence_score = 0

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            c = corners[0][0]
            marker_cx = int((c[0][0] + c[2][0]) / 2)
            marker_cy = int((c[0][1] + c[2][1]) / 2)

            # AI Calculations
            error_x = screen_center[0] - marker_cx
            error_y = screen_center[1] - marker_cy
            thrust_x = pid_x.compute(error_x)
            thrust_y = pid_y.compute(error_y)

            # Metrics
            tilt = calculate_angle(c[0], c[1])
            dist_factor = np.linalg.norm(c[0] - c[1])

            # Confidence Logic
            raw_score = 100 - (abs(error_x) / 5 + abs(error_y) / 5 + abs(tilt) * 2)
            confidence_score = np.clip(raw_score, 0, 100)

            # Draw AI Vectors
            draw_vector_arrow(frame, screen_center, -thrust_x, 0)
            draw_vector_arrow(frame, screen_center, 0, -thrust_y)
            cv2.line(frame, screen_center, (marker_cx, marker_cy), (255, 255, 0), 1)

            # --- LOGIC FOR STATUS MESSAGES ---
            if abs(tilt) > 15:
                status_msg = "WARNING: LEVEL OUT"
                status_color = (0, 0, 255)  # Red

            elif dist_factor > 180:
                status_msg = "DOCKED - CHARGING"
                status_color = (0, 255, 0)  # Green
                confidence_score = 100
                battery_level = min(100, battery_level + 0.5)
                draw_docking_overlay(frame)

            elif abs(error_x) > 40 or abs(error_y) > 40:
                status_msg = "ALIGNING..."
                status_color = (0, 165, 255)  # Orange

            else:
                status_msg = "APPROACHING TARGET"
                status_color = (255, 255, 0)  # Cyan

        # Battery Drain
        if status_msg != "DOCKED - CHARGING":
            if time.time() - last_drain_time >= 5:
                battery_level = max(0, battery_level - 1)
                last_drain_time = time.time()

        draw_sci_fi_hud(
            frame, battery_level, confidence_score, start_time, status_msg, status_color
        )

        cv2.imshow("AI Drone HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_advanced_docking()

import cv2
import cv2.aruco as aruco
import numpy as np
import time


def calculate_angle(p1, p2):
    """Calculates the angle (tilt) between two points."""
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    return np.degrees(angle_rad)


def draw_battery(frame, level):
    """Draws a battery bar on the screen."""
    # Color logic: Green if high, Red if low
    color = (0, 255, 0) if level > 20 else (0, 0, 255)

    # Draw outline
    cv2.rectangle(
        frame, (20, frame.shape[0] - 50), (220, frame.shape[0] - 20), (255, 255, 255), 2
    )
    # Draw fill based on level
    fill_width = int(200 * (level / 100))
    cv2.rectangle(
        frame,
        (22, frame.shape[0] - 48),
        (22 + fill_width, frame.shape[0] - 22),
        color,
        -1,
    )

    # Text
    cv2.putText(
        frame,
        f"BATTERY: {int(level)}%",
        (240, frame.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def start_advanced_docking():
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # --- SIMULATION VARIABLES ---
    battery_level = 100.0
    start_time = time.time()
    is_charging = False

    # NEW: Drain battery 1% every 10 seconds
    last_drain_time = time.time()

    print("Advanced System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        screen_center_x = frame.shape[1] // 2
        screen_center_y = frame.shape[0] // 2

        # Draw HUD Crosshair
        cv2.line(
            frame,
            (screen_center_x - 20, screen_center_y),
            (screen_center_x + 20, screen_center_y),
            (0, 255, 0),
            1,
        )
        cv2.line(
            frame,
            (screen_center_x, screen_center_y - 20),
            (screen_center_x, screen_center_y + 20),
            (0, 255, 0),
            1,
        )

        docking_status = "SEARCHING"
        color_status = (0, 165, 255)  # Orange

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Get corners of the first marker
            c = corners[0][0]

            # 1. Center Calculation
            marker_center_x = int((c[0][0] + c[2][0]) / 2)
            marker_center_y = int((c[0][1] + c[2][1]) / 2)
            cv2.line(
                frame,
                (screen_center_x, screen_center_y),
                (marker_center_x, marker_center_y),
                (255, 255, 0),
                1,
            )

            # 2. Size (Distance) Calculation
            marker_width = np.linalg.norm(c[0] - c[1])

            # 3. Angle (Tilt)
            tilt_angle = calculate_angle(c[0], c[1])

            # --- LOGIC TREE ---
            if abs(tilt_angle) > 10:
                docking_status = "WARNING: LEVEL OUT!"
                color_status = (0, 0, 255)
                is_charging = False

            elif abs(marker_center_x - screen_center_x) > 50:
                docking_status = "ALIGNING..."
                color_status = (0, 255, 255)
                is_charging = False

            elif marker_width > 180:
                docking_status = "DOCKED - CHARGING"
                color_status = (0, 255, 0)
                is_charging = True

            else:
                docking_status = "APPROACHING"
                color_status = (0, 255, 255)
                is_charging = False

        else:
            is_charging = False

        # --- SIMULATION UPDATES ---
        current_time = time.time()

        if is_charging:
            if battery_level < 100:
                battery_level += 0.5  # Charging speed
        else:
            # Drain 1% every 10 seconds
            if current_time - last_drain_time >= 10:
                if battery_level > 0:
                    battery_level -= 1
                last_drain_time = current_time

        # --- DRAW UI ---
        cv2.putText(
            frame,
            docking_status,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_status,
            3,
        )

        draw_battery(frame, battery_level)

        elapsed = int(time.time() - start_time)
        cv2.putText(
            frame,
            f"T+: {elapsed}s",
            (frame.shape[1] - 150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Advanced Drone Charger", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_advanced_docking()

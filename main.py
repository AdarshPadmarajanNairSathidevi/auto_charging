import cv2
import cv2.aruco as aruco
import numpy as np


def start_docking():
    cap = cv2.VideoCapture(0)

    # Load the dictionary that was used to generate the markers.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    print("System active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        # Get the center of the screen
        screen_center_x = frame.shape[1] // 2
        screen_center_y = frame.shape[0] // 2

        # Draw a crosshair in the middle of the screen (The "Drone's Target")
        cv2.line(
            frame,
            (screen_center_x, 0),
            (screen_center_x, frame.shape[0]),
            (255, 0, 0),
            1,
        )
        cv2.line(
            frame,
            (0, screen_center_y),
            (frame.shape[1], screen_center_y),
            (255, 0, 0),
            1,
        )

        if ids is not None:
            # --- MARKER FOUND ---
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 1. GET COORDINATES
            # 'c' is the list of 4 corner points of the marker
            c = corners[0][0]

            # Calculate the center of the marker
            marker_center_x = int((c[0][0] + c[2][0]) / 2)
            marker_center_y = int((c[0][1] + c[2][1]) / 2)

            # 2. CALCULATE "APPARENT SIZE" (Proxy for Distance)
            # We measure the width from Top-Left corner (c[0]) to Top-Right corner (c[1])
            # If this number is BIG, the marker is CLOSE.
            marker_width_pixels = np.linalg.norm(c[0] - c[1])

            # Draw a line from screen center to marker center (Guidance Line)
            cv2.line(
                frame,
                (screen_center_x, screen_center_y),
                (marker_center_x, marker_center_y),
                (0, 255, 255),
                2,
            )

            # --- DOCKING LOGIC ---

            # Tolerance: How precise must we be? (pixels)
            tolerance = 50

            # Check Alignment (Left/Right)
            if marker_center_x < screen_center_x - tolerance:
                cv2.putText(
                    frame,
                    "<< MOVE LEFT",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

            elif marker_center_x > screen_center_x + tolerance:
                cv2.putText(
                    frame,
                    "MOVE RIGHT >>",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

            else:
                # WE ARE ALIGNED! Now check Distance.

                # Threshold: If marker width is > 180 pixels, we consider it "Docked"
                # You can change 180 to be larger (closer) or smaller (farther)
                if marker_width_pixels > 180:
                    # SUCCESS STATE
                    cv2.putText(
                        frame,
                        "DOCKED SUCCESSFULLY!",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                    )
                    # Draw a nice circle to show success
                    cv2.circle(
                        frame, (screen_center_x, screen_center_y), 50, (0, 255, 0), 5
                    )
                else:
                    # ALIGNED BUT TOO FAR
                    cv2.putText(
                        frame,
                        "APPROACHING...",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3,
                    )
                    # Show distance estimation (optional)
                    cv2.putText(
                        frame,
                        f"Width: {int(marker_width_pixels)}px",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

        else:
            cv2.putText(
                frame,
                "SEARCHING...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Drone Docking System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_docking()
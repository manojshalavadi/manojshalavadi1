import cv2

for i in range(5):
    print(f"\nTesting camera index {i}")
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print("Not opened")
        continue

    ret, frame = cap.read()
    print("Opened:", cap.isOpened())
    print("Frame received:", ret)

    if ret:
        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cap.release()
        break

    cap.release()
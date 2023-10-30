from ultralytics import YOLO
import cv2

import torch
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available. Using CPU.")

if torch.cuda.current_device():
    print("Yes1")
else:
    print("No1")

num_gpus = torch.cuda.device_count()

torch.cuda.set_device(0)

current_gpu = torch.cuda.current_device()
print(f"Current GPU: {current_gpu}")


from sort.sort import *
from util import get_car, read_license_plate, write_csv

cv2.namedWindow('Detected Video', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Detected Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt').to("cuda")
license_plate_detector = YOLO('./models/best.pt').to("cuda")

# load video
cap = cv2.VideoCapture('./videos/IMG_4424.mp4')

vehicles = [2, 3, 5, 7]

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        frame_tensor = torch.from_numpy(frame).to('cuda')
        results[frame_nmr] = {}
        detections = coco_model(frame)[0].to('cpu')
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Vẽ bounding boxes xung quanh các đối tượng đã phát hiện
        for box in detections_:
            x1, y1, x2, y2, _ = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Hiển thị video trong chế độ full màn hình
        cv2.imshow('Detected Video', frame)

        # Đợi một khoảng thời gian ngắn (vd: 10 ms) và kiểm tra nút bấm "q" để thoát
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detection license plate
        license_plates = license_plate_detector(frame)[0].to('cpu')
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # progress license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 90, 255, cv2.THRESH_BINARY_INV)

            cv2.imshow('origin_crop', license_plate_crop)
            cv2.imshow('threshold_crop', license_plate_crop_thresh)

            # read license plate
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            print('license plate', license_plate_text)
            print('license plate score', license_plate_text_score)
            print(coco_model.device)  # In ra thiết bị của mô hình YOLO
            print(license_plate_detector.device)  # In ra thiết bị của mô hình nhận dạng biển số
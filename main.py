from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv


cv2.namedWindow('Detected Video', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Detected Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt')
license_plate_detector = YOLO('./models/best.pt')

# load video
cap = cv2.VideoCapture('./videos/IMG_4424.MP4')

vehicles = [2, 3, 5, 7]

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        results[frame_nmr] = {}
        detections = coco_model(frame)[0]
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
        license_plates = license_plate_detector(frame)[0]
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

            if license_plate_text is not None :
                print('license plate', license_plate_text)
                print('license plate score', license_plate_text_score)
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
                print(results)
            write_csv(results, './test.csv')
            if os.path.exists('./test.csv'):
                print(True)
            else:
                print(False)



# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import *
# from util import get_car, read_license_plate
#
# # Khởi tạo cửa sổ hiển thị
# cv2.namedWindow('Detected Image', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('Detected Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
# mot_tracker = Sort()
#
# # Load các mô hình
# coco_model = YOLO('./yolov8n.pt')
# license_plate_detector = YOLO('./models/best.pt')
#
# # Đường dẫn ảnh cần xử lý
# image_path = './images/img1.png'
#
# # Đọc ảnh
# frame = cv2.imread(image_path)
#
# # Lấy danh sách các frame (1 frame ở đây tương đương với 1 bức ảnh)
# frames = [frame]  # Thêm frame ban đầu vào danh sách
#
# # Thực hiện xử lý cho từng frame
# for frame in frames:
#     vehicles = [2, 3, 5, 7]
#
#     results = {}
#     detections_ = []
#
#     # Detect các đối tượng trên frame
#     detections = coco_model(frame)[0]
#
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])
#
#     # Vẽ bounding boxes xung quanh các đối tượng đã phát hiện
#     for box in detections_:
#         x1, y1, x2, y2, _ = box
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#
#     # Hiển thị ảnh trong chế độ full màn hình
#     cv2.imshow('Detected Image', frame)
#
#     # Track các đối tượng
#     track_ids = mot_tracker.update(np.asarray(detections_))
#
#     # Detect biển số xe
#     license_plates = license_plate_detector(frame)[0]
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate
#
#         # Gán biển số xe cho xe
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#
#         # Cắt biển số xe
#         license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#
#         # Xử lý biển số xe
#         license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#         _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 180, 255, cv2.THRESH_BINARY_INV)
#
#         cv2.imshow('origin_crop', license_plate_crop)
#         cv2.imshow('threshold_crop', license_plate_crop_thresh)
#
#         # Đọc biển số xe
#         license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#         print('license plate', license_plate_text)
#         print('license plate score', license_plate_text_score)
#
#
# if os.path.exists('./test.csv'):
#     print("True")
# else: print("False")


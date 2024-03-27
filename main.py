import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model

# Predict with the model
results = model('short.mov')  # predict on an image
print(results)
results[0].show()


# def video_pose(filename, out_filename):
#     # Open the input video file and extract its properties
#     cap = cv2.VideoCapture(filename)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     # Create VideoWriter object
#     out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
#     #  Processing a video file frame by frame
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             pred = make_pose_prediction(model, frame)
#             plot_pose_prediction(frame, pred, show_bbox=False)
#             out.write(frame)
#             cv2.imshow('Pose estimation', frame)
#         else:
#             break
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     # Release VideoCapture and VideoWriter
#     cap.release()
#     out.release()
#     # Close all frames and video windows
#     cv2.destroyAllWindows()
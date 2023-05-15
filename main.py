from detection import detect_video
from model import get_trained_model, train_model
from utils import normalize_filenames, count_classes

### Configurations ###

# Verbose during prediction
VERBOSE = False
# Scaling percentage of original frame
SCALE_PERCENT = 50
# Confidence threshold for detection
CONF = 0.60
# Input path
INPUT_PATH = 'datasets/video/vehicle-counting.mp4'
# Output path
OUTPUT_PATH = "C:\\Users\\Dell\\PycharmProjects\\yoloDectection\\runs\\detect_video"
# Video name
VIDEO_NAME = 'test.mp4'
# Classes to detect
CLASS = None # [2, 3, 5, 6, 7]


if __name__ == '__main__':
    count_classes('datasets/vehicle_dataset/', skip_repeats=False)
    a = 0
    normalize_filenames('datasets/vehicle_dataset/')
    model = train_model('datasets/vehicle_dataset.yaml', 'trained_models/yolov8n.pt', 3)
    # model = get_trained_model('trained_models/custom_yolov8s_epoch10.pt')
    # model = get_trained_model('trained_models/yolov8x.pt')
    detect_video(model,
                 INPUT_PATH,
                 OUTPUT_PATH,
                 VIDEO_NAME,
                 class_id=CLASS,
                 conf_threshold=CONF,
                 scale_percent=SCALE_PERCENT,
                 verbose=VERBOSE)

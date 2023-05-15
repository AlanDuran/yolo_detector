import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from video import get_video, resize_frame


def detect_video(model, path, output_path, video_name, conf_threshold=0.7,
                 class_id=None, scale_percent=100, verbose=False):
    output_video_path = os.path.join(output_path, video_name)
    video, output_video = get_video(path, output_video_path, scale_percent)

    # getting names from classes
    dict_classes = model.model.names

    # Auxiliary variables
    counter_keys = class_id if class_id is not None else dict_classes.keys()
    vehicles_counter_in = dict.fromkeys(counter_keys, 0)
    vehicles_counter_out = dict.fromkeys(counter_keys, 0)
    frames_list = []
    cy_line = int(1200 * scale_percent / 100)
    cx_line = int(2000 * scale_percent / 100)
    offset = int(15 * scale_percent / 100)
    counter_in = 0
    counter_out = 0
    in_ids = []
    out_ids = []

    # Executing Recognition
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):

        # reading frame from video
        _, frame = video.read()

        # Applying resizing of read frame
        frame = resize_frame(frame, scale_percent)

        results = model.track(frame, conf=conf_threshold, classes=class_id,
                              verbose=verbose, persist=True)
        # Getting predictions
        # y_hat = model.predict(frame, conf=conf_threshold, classes=class_id,
        #                     verbose=verbose)

        # Getting the bounding boxes, confidence and classes of the
        # objects in the current frame.
        conf = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Storing the above information in a dataframe
        boxes_data = results[0].cpu().numpy().boxes.data

        if boxes_data.shape[1] == 6:
            boxes_data = np.zeros((0, 7))
        positions_frame = pd.DataFrame(boxes_data,
                                       columns=['xmin', 'ymin', 'xmax', 'ymax',
                                                'id', 'conf', 'class'])

        # Translating the numeric class labels to text
        labels = [dict_classes[i] for i in classes]

        # Drawing transition line for in\out vehicles counting
        cv2.line(frame, (0, cy_line),
                 (int(4500 * scale_percent / 100), cy_line), (255, 255, 0), 8)

        # For each vehicle, draw the bounding-box
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, id, confidence, category, = row[1].astype(
                'int')

            # Calculating the center of the bounding-box
            center_x, center_y = int(((xmax + xmin)) / 2), int(
                (ymax + ymin) / 2)

            # drawing center and bounding-box of vehicle in the given frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0),
                          5)  # box
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0),
                       -1)  # center of box

            # Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=frame,
                        text=labels[ix] + ' (' + str(id) + ') - ' + str(
                            np.round(conf[ix], 2)),
                        org=(xmin, ymin - 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)

            # Checking if the center of recognized vehicle is in the area given
            # by the transition line + offset and transition line - offset
            if (center_y < (cy_line + offset)) and (
                    center_y > (cy_line - offset)):
                if (center_x >= 0) and (center_x <= cx_line) and\
                        (id not in in_ids):
                    counter_in += 1
                    vehicles_counter_in[category] += 1
                    in_ids.append(id)
                elif (center_x > cx_line) and id not in out_ids:
                    counter_out += 1
                    vehicles_counter_out[category] += 1
                    out_ids.append(id)

        # updating the counting type of vehicle
        counter_in_plt = [f'{dict_classes[k]}: {i}' for k, i in
                          vehicles_counter_in.items()]
        counter_out_plt = [f'{dict_classes[k]}: {i}' for k, i in
                           vehicles_counter_out.items()]

        # drawing the number of vehicles in\out
        cv2.putText(img=frame, text='N. vehicles In',
                    org=(30, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0), thickness=1)

        cv2.putText(img=frame, text='N. vehicles Out',
                    org=(int(2800 * scale_percent / 100), 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(255, 255, 0), thickness=1)

        # drawing the counting of type of vehicles in the corners of frame
        xt = 40
        for txt in range(len(counter_in_plt)):
            xt += 30
            cv2.putText(img=frame, text=counter_in_plt[txt],
                        org=(30, xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(255, 255, 0), thickness=1)

            cv2.putText(img=frame, text=counter_out_plt[txt],
                        org=(int(2800 * scale_percent / 100), xt),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(255, 255, 0), thickness=1)

        # drawing the number of vehicles in\out
        cv2.putText(img=frame, text=f'In:{counter_in}',
                    org=(int(1820 * scale_percent / 100), cy_line + 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(255, 255, 0), thickness=2)

        cv2.putText(img=frame, text=f'Out:{counter_out}',
                    org=(int(1800 * scale_percent / 100), cy_line - 40),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(255, 255, 0), thickness=2)

        if verbose:
            print(counter_in, counter_out)
        # Saving frames in a list
        frames_list.append(frame)
        # saving transformed frames in a output video formaat
        output_video.write(frame)

    # Releasing the video
    output_video.release()

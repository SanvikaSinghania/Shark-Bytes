import cv2
import numpy as np
import glob
import os
from numpy import random
from google.cloud import vision_v1 as vision
import pandas as pd
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Add_your_google_cloud_credentials_here.json'
client = vision.ImageAnnotatorClient()


win_name = 'Camera Matching'
MIN_MATCH = 10
images = glob.glob('*.JPG')  # all jpg images in the folder could be displayed
currentImage = random.choice([0, 1, 2, 3, 4, 5])
replaceImg = cv2.imread(images[currentImage])
rows, cols, ch = replaceImg.shape
pts1 = np.float32([[0, 0], [0, rows], [(cols), (rows)], [cols, 0]])
maskThreshold = 10
# Detector used runs ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.ORB_create(1000)
# Flann for approximating the nearest neighbour
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Start of video capture and setting the frame size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while cap.isOpened():
    ret, frame = cap.read()

    success, encoded_image = cv2.imencode('.jpg', frame)
    content = encoded_image.tobytes()
    image = vision.types.Image(content=content)
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations
    df = pd.DataFrame(columns=['name', 'score'])
    for obj in localized_object_annotations:
        df = df.append(
            dict(
                name=obj.name,
                score=obj.score
            ),
            ignore_index=True)
    print(df)

    img_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(img_temp)

    img1 = None
    for obj in localized_object_annotations:
        if(('shark' in obj.name) or ('Shark' in obj.name)):
            r, g, b = 255, 255, 255
            width, height = pillow_image.size

            pillow_image = pillow_image.crop((obj.bounding_poly.normalized_vertices[0].x *
                                              width, obj.bounding_poly.normalized_vertices[0].y * height,
                                              obj.bounding_poly.normalized_vertices[2].x *
                                              width, obj.bounding_poly.normalized_vertices[2].y * height))
            img1 = np.array(pillow_image)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            #img1 = cvtColor(np.asarray(pillow_image), COLOR_BGR2RGB)

    if img1 is None:
        res = frame
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)
        ratio = 0.75
        good_matches = [m[0] for m in matches
                        if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        matchesMask = np.zeros(len(good_matches)).tolist()
        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            #mask = cv2.erode(mask, (3, 3))
            #mask = cv2.dilate(mask, (3, 3))
            if mask.sum() > MIN_MATCH:
                matchesMask = mask.ravel().tolist()
                h, w, = img1.shape[:2]
                pts = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                dst = cv2.perspectiveTransform(pts, mtrx)
                dst = cv2.getPerspectiveTransform(pts1, dst)
                rows, cols, ch = frame.shape
                distance = cv2.warpPerspective(replaceImg, dst, (cols, rows))
                rt, mk = cv2.threshold(cv2.cvtColor(distance, cv2.COLOR_BGR2GRAY),
                                       maskThreshold, 1, cv2.THRESH_BINARY_INV)
                mk = cv2.erode(mk, (3, 3))
                mk = cv2.dilate(mk, (3, 3))

                for c in range(0, 3):
                    frame[:, :, c] = distance[:, :, c]*(1-mk[:, :]) + frame[:, :, c]*mk[:, :]
    cv2.imshow('img', frame)
    # Wait for the key
    key = cv2.waitKey(1)
    # decide the action based on the key value (quit, zoom, change image)
    if key == ord('q'):  # quit
        print('Quit')
        break
cap.release()
cv2.destroyAllWindows()

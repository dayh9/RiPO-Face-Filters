import numpy as np
import dlib
import cv2
from imutils import face_utils
import argparse
import os
import imutils
import math


def drawAllPoints(img, pts):
    overlay = img.copy()
    # drawing face landmarks
    for i in range(len(pts)):
        cv2.circle(overlay, (pts[i][0], pts[i][1]), 1, (0, 0, 250), 2)

    return overlay

def scarryFace(img, pts): # showing eyes and mouth only
    overlay = img.copy()

    eyelayer = np.zeros(overlay.shape, dtype='uint8')
    eye_mask = eyelayer.copy()
    eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_BGR2GRAY)
    # definiowanie punktów wycinanych elementów twarzy
    right_eye = pts[36:42]
    left_eye = pts[42:48]
    mouth = pts[48:60]
    cv2.fillPoly(eye_mask, [left_eye], 255)
    cv2.fillPoly(eye_mask, [right_eye], 255)
    cv2.fillPoly(eye_mask, [mouth], 255)
    eyelayer = cv2.bitwise_and(overlay, overlay, mask=eye_mask)  # nakładanie maski

    return eyelayer

def blurFace(img, pts):  # blurring face
    overlay = img.copy()

    face_shape = pts[0:16]
    face_shape2 = np.array(pts[78])
    face_shape2 = np.vstack([face_shape2, [pts[74], pts[79], pts[73], pts[72], pts[80], pts[71], pts[70], pts[69],
                                           pts[68], pts[76], pts[75], pts[77]]])
    face_shape = np.append(face_shape, face_shape2, 0)

    insensitivity = (overlay.shape[0] + overlay.shape[1]) / 2
    insensitivity = math.floor(insensitivity / 20)
    if insensitivity % 2 == 0:
        insensitivity = insensitivity + 1
    blurred_image = cv2.GaussianBlur(overlay, (insensitivity, insensitivity), insensitivity - 1)

    mask = np.zeros(overlay.shape, dtype=np.uint8)
    channel_count = overlay.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, np.int32([face_shape]), ignore_mask_color)

    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask

    final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(overlay, mask_inverse)

    return final_image


def FaceFilter(frame):
    op = frame.copy()
    gray = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY)

    bounding_boxes = face_detector(gray,
                                   0)


    if bounding_boxes:
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)
            #op = drawAllPoints(op, face_landmark_points)
            #op = blurFace(op, face_landmark_points)
            op = scarryFace(op, face_landmark_points)

        return op
    else:
        return frame


def video(src=0):
    cap = cv2.VideoCapture(src)

    if args['save']:
        if os.path.isfile(args['save'] + '.avi'):
            os.remove(args['save'] + '.avi')
        out = cv2.VideoWriter(args['save'] + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened:
        _, frame = cap.read()
        output_frame = FaceFilter(frame)

        if args['save']:
            out.write(output_frame)
        cv2.imshow("Face Filter", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args['save']:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def image(source):
    if os.path.isfile(source):
        img = cv2.imread(source)
        output_frame = FaceFilter(img)
        resized_img = imutils.resize(output_frame, height=900)
        cv2.imshow("Face Filter", resized_img)
        if args['save']:
            if os.path.isfile(args['save'] + '.png'):
                os.remove(args['save'] + '.png')
            cv2.imwrite(args['save'] + '.png', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("File not found :( ")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="Path to video file")
    ap.add_argument("-i", "--image", required=False, help="Path to image")
    ap.add_argument("-d", "--dat", required=False, help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-s", "--save", required=False, help='Enter the file name to save')
    args = vars(ap.parse_args())

    if args['dat']:
        dataFile = args['dat']

    else:
        dataFile = "shape_predictor_81_face_landmarks.dat"

    color = (0, 0, 0)
    thickness = 2
    face_detector = dlib.get_frontal_face_detector()
    lndMrkDetector = dlib.shape_predictor(dataFile)

    if args['image']:
        image(args['image'])

    if args['video'] and args['video'] != 'webcam':
        if os.path.isfile(args['video']):
            video(args['video'])

        else:
            print("File not found :( ")

    elif args['video'] == 'webcam':
        video(0)

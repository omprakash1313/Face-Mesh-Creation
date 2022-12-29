import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # input for the source
pTime = 0  # current time--> initial time =0

mpDraw = mp.solutions.drawing_utils    # creating object for mediapipe
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# features and specification of the face 'i.e' num of face + thickness+radius of each circle
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()  # capturing for image or source
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # converting from BGR to RGB color bec. it supports only RGB
    results = faceMesh.process(imgRGB)   # passing the RGB image to the function
    if results.multi_face_landmarks:   # if the points desired are found 'i.e' landmarks found then move forward
        for faceLms in results.multi_face_landmarks:
            # sending parameters if the landmarks are found with specification
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            for Id, lm in enumerate(faceLms.landmark):
                # iterating to each point and find the coordinate of the face in form X and Y
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(Id, x, y)
    # calculation of frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # putting frame rate on the text or the video
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)  # input of fps and the color of the fps rate display (20,70)--> size of the window
    cv2.imshow("Image", img)  # calling of the object imshow to display the output
    cv2.waitKey(1)
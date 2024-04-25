import cv2
import pickle

width, height = 160,200

try:
    with open('IDS', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x ,y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open('IDS', 'wb') as f:
        pickle.dump(posList, f)

while True:
    img = cv2.imread('alert /Screenshot 2023-10-30 at 8.33.35 PM.png')
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2) 

    cv2.imshow("Images", img)
    cv2.setMouseCallback("Images", mouseClick)
    cv2.waitKey(1)

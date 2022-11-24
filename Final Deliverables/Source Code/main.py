import cv2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    camera = cv2.VideoCapture("vid.mkv")
    while True:
        data,frame = camera.read()
        result = frame.copy()
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours , hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        colour = (255, 0, 0)
        thickness = 1
        i = 0
        for cntr in contours:
            x1, y1, w, h = cv2.boundingRect(cntr)
            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(result, (x1, y1), (x2, y2), colour, thickness)
            print("Object:", i + 1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
            i += 1

        cv2.imshow("img",result)
        cv2.waitKey(5)


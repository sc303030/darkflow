import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image


font = ImageFont.truetype("fonts/gulim.ttc", 20)
img = np.full((200, 300, 3), (255, 255, 255), np.uint8)
img = Image.fromarray(img)
draw = ImageDraw.Draw(img)



options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.2,
    'gpu': 1.0
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('nba_sample3.avi')
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# capture.set(cv2.CAP_PROP_FPS, int(60))

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(width), int(height)))

prev_time = 0
FPS = 10

while True:
    # stime = time.time()
    ret, frame = capture.read()
    current_time = time.time() - prev_time
    if(ret is True) and (current_time > 1./ FPS):
        prev_time = time.time()
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 2)
            draw.text((30, 50), text, font=font, fill=(0, 0, 0))
            img = np.array(img)
            text_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)
            text_width, text_height = text_size[0][0], text_size[0][1]
            text_resize = text_width*0.7
            cv2.rectangle(frame, pt1=(result['topleft']['x']-1, result['topleft']['y'] - text_height), pt2=(result['topleft']['x']-30 + text_width, result['topleft']['y']), color=color, thickness=-1)
            frame = cv2.putText(
                frame, text, (result['topleft']['x'], result['topleft']['y']-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.imshow('frame', frame)
        out.write(frame)
        print('FPS {:.1f}'.format(1 / (time.time() - prev_time)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
cv2.destroyAllWindows()

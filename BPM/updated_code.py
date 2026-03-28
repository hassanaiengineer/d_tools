import numpy as np
import cv2
from scipy.signal import find_peaks
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time

# Video and detection settings
realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoFrameRate = 30

video_path = 'videos/Video 5.mp4'  
webcam = cv2.VideoCapture(video_path)
detector = FaceDetector()

webcam.set(3, realWidth)
webcam.set(4, realHeight)

plotY = LivePlot(realWidth, realHeight, [50, 120], invert=True)

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)

buffer_size = videoFrameRate * 10  
color_buffer = []
bpmBufferSize = 30  
bpmBuffer = np.zeros((bpmBufferSize))
bpmBufferIndex = 0

def moving_average(data, n=5):
    if len(data) < n:
        return data
    return np.convolve(data, np.ones(n) / n, mode='valid')

def process_color_buffer(fps):
    if len(color_buffer) == buffer_size:
        green_channel = [c[1] for c in color_buffer]
        detrended = np.array(green_channel) - np.mean(green_channel)

        peaks, _ = find_peaks(detrended, distance=fps//2)

        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            heart_rates = 60 * fps / peak_intervals
            current_hr = np.median(heart_rates)
            return current_hr
    return None

ptime = 0
i = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()

    fps = 1 / (ftime - ptime) if ptime != 0 else 0
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)

    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
        
        mean_color = np.mean(detectionFrame, axis=(0, 1))
        color_buffer.append(mean_color)

        if len(color_buffer) > buffer_size:
            color_buffer.pop(0)

        current_bpm = process_color_buffer(fps)

        if current_bpm is not None:
            bpmBuffer[bpmBufferIndex] = current_bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        bpm_value = moving_average(bpmBuffer).mean() if i > bpmBufferSize else bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        if i >= bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', bpmTextLocation, scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

        i += 1

        imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

webcam.release()
cv2.destroyAllWindows()

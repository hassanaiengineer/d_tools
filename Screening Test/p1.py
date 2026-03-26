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

# Video path for input (change this to your video source)
video_path = 'videos/Sample_video_to_on.mp4'  # Update with your video path
webcam = cv2.VideoCapture(video_path)

# Initialize Face Detector from cvzone
detector = FaceDetector()

# Video capture settings
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Plotting for BPM
plotY = LivePlot(realWidth, realHeight, [50, 120], invert=True)

# Display settings
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)

# Parameters for heart rate calculation
buffer_size = videoFrameRate * 10  # Buffer for 10 seconds
color_buffer = []
bpmBufferSize = 30  # Increased buffer size for averaging BPM
bpmBuffer = np.zeros((bpmBufferSize))
bpmBufferIndex = 0

# Moving average function to smooth BPM data
def moving_average(data, n=5):
    if len(data) < n:
        return data
    return np.convolve(data, np.ones(n) / n, mode='valid')

# Function to process the green channel and calculate BPM
def process_color_buffer(fps):
    if len(color_buffer) == buffer_size:
        green_channel = [c[1] for c in color_buffer]
        detrended = np.array(green_channel) - np.mean(green_channel)

        # Simple peak detection
        peaks, _ = find_peaks(detrended, distance=fps // 2)

        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            heart_rates = 60 * fps / peak_intervals
            current_hr = np.median(heart_rates)
            return current_hr
    return None

# Variables for FPS calculation
ptime = 0
i = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Detect faces in the frame using cvzone
    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()

    # FPS calculation
    fps = 1 / (ftime - ptime) if ptime != 0 else 0
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)

    if bboxs:
        # Face region detection using cvzone
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        # Process the face region (ROI) to calculate mean color
        mean_color = np.mean(detectionFrame, axis=(0, 1))
        color_buffer.append(mean_color)

        if len(color_buffer) > buffer_size:
            color_buffer.pop(0)

        # Calculate BPM using the green channel data
        current_bpm = process_color_buffer(fps)

        if current_bpm is not None:
            bpmBuffer[bpmBufferIndex] = current_bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Apply moving average filter for smoother BPM data
        bpm_value = moving_average(bpmBuffer).mean() if i > bpmBufferSize else bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        # Ensure BPM value is displayed after sufficient data has been collected
        if i >= bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', bpmTextLocation, scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

        i += 1

        # Stack the images with both the frame and the BPM plot
        outputFrame_show = cv2.resize(detectionFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

webcam.release()
cv2.destroyAllWindows()

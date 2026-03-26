import numpy as np
import cv2
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft
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

# Provide the path of your video
video_path = 'videos/Video 5.mp4'  # Change this for each video

detector = FaceDetector()

plotY = LivePlot(realWidth, realHeight, [50, 120], invert=True)

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)

buffer_size = videoFrameRate * 10
bpmBufferSize = 30

# Preprocessing: Low-pass and high-pass filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half of the sampling rate
    low = lowcut / nyquist
    high = highcut / nyquist
    if low >= 1 or high >= 1 or low <= 0 or high <= 0:
        raise ValueError("Lowcut and Highcut must be between 0 and Nyquist frequency.")
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply bandpass filter to data
def apply_bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30.0, order=4):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    except ValueError as e:
        print(f"Error applying bandpass filter: {e}")
        return data  # If filter fails, return the unfiltered data

# Moving average function to smooth BPM data
def moving_average(data, n=5):
    if len(data) < n:
        return data
    return np.convolve(data, np.ones(n) / n, mode='valid')

# Process color buffer using FFT for frequency detection
def process_color_buffer(color_buffer, fps):
    if len(color_buffer) == buffer_size:
        green_channel = np.array([c[1] for c in color_buffer])

        # Apply detrending
        detrended = green_channel - np.mean(green_channel)

        # Apply bandpass filter
        filtered_signal = apply_bandpass_filter(detrended, lowcut=0.75, highcut=2.5, fs=fps)

        # FFT for frequency domain analysis
        fft_signal = np.abs(fft(filtered_signal))
        freqs = np.fft.fftfreq(len(filtered_signal), d=1/fps)

        # Find peak frequency within heart rate range (0.75 Hz to 2.5 Hz)
        valid_freqs = (freqs >= 0.75) & (freqs <= 2.5)
        peak_freq = freqs[valid_freqs][np.argmax(fft_signal[valid_freqs])]

        bpm = peak_freq * 60  # Convert Hz to BPM
        return bpm
    return None

# Function to reset buffers and variables
def reset_buffers():
    global color_buffer, bpmBuffer, bpmBufferIndex, i
    color_buffer = []  # Reset color buffer
    bpmBuffer = np.zeros((bpmBufferSize))  # Reset BPM buffer
    bpmBufferIndex = 0
    i = 0  # Reset frame counter

# Initialize webcam for the current video
webcam = cv2.VideoCapture(video_path)
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Reset buffers for each new test
reset_buffers()

ptime = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

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

        # Extract mean color from face ROI
        mean_color = np.mean(detectionFrame, axis=(0, 1))
        color_buffer.append(mean_color)

        if len(color_buffer) > buffer_size:
            color_buffer.pop(0)

        # Calculate BPM using FFT-based analysis
        current_bpm = process_color_buffer(color_buffer, fps)

        if current_bpm is not None:
            bpmBuffer[bpmBufferIndex] = current_bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Apply moving average filter for smoother BPM data
        bpm_value = moving_average(bpmBuffer).mean() if i > bpmBufferSize else bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        # Display BPM after sufficient data has been collected
        if i >= bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', bpmTextLocation, scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

        i += 1

        # Display the stacked images with both the frame and the BPM plot
        imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)

webcam.release()
cv2.destroyAllWindows()

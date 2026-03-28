import numpy as np
import cv2
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fftpack import fft
import time

# Video and detection settings
realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoFrameRate = 30

# Provide the path of your video
video_path = 'videos\Sample_video_to_on.mp4'  # Change this for each video
webcam = cv2.VideoCapture(video_path)

# Preprocessing: Low-pass and high-pass filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.75, highcut=2.5, fs=30.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def moving_average(data, n=5):
    if len(data) < n:
        return data
    return np.convolve(data, np.ones(n) / n, mode='valid')

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

# Initialize webcam
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

buffer_size = videoFrameRate * 10
color_buffer = []
bpmBufferSize = 30
bpmBuffer = np.zeros((bpmBufferSize))
bpmBufferIndex = 0

ptime = 0
i = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    ftime = time.time()
    fps = 1 / (ftime - ptime) if ptime != 0 else 0
    ptime = ftime

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Focus on the forehead region (upper part of the face)
        forehead_region = frame[y:y + h//4, x:x + w]
        cv2.imshow('Forehead Region', forehead_region)

        # Extract mean color from forehead ROI
        mean_color = np.mean(forehead_region, axis=(0, 1))
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

        cv2.putText(frame, f'BPM: {bpm_value:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        i += 1

    cv2.imshow("Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

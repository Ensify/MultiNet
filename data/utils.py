import librosa
import soundfile as sf
import cv2
import numpy as np
import os
import random

def get_random_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = random.sample(range(total_frames), num_frames)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    max_area = 0
    max_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            max_face = image[y:y+h, x:x+w]

    return max_face

def extract_audio_from_video(video_path,audio_path):
   audio , sr = librosa.load(video_path)
   sf.write(audio_path,audio,sr)

def makeDataset(dataset, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for emotion in dataset.keys():
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(os.path.join(emotion_dir, "video"))
            os.makedirs(os.path.join(emotion_dir, "audio"))
        for file in dataset[emotion]:
            video_dir = os.path.join(emotion_dir, "video", os.path.splitext(os.path.basename(file))[0])
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            # Extract random frames
            frames = get_random_frames(file)
            frame_count = 0
            for frame in frames:
                cropped_faces = [detect_and_crop_faces(frame)]
                if cropped_faces[0] is not None:
                    for i, face in enumerate(cropped_faces):
                        face_path = os.path.join(video_dir, f"frame{frame_count}_face{i}.jpg")
                        cv2.imwrite(face_path, face)
                frame_count += 1

            # Extract audio
            audio_output_path = os.path.join(emotion_dir, "audio", os.path.splitext(os.path.basename(file))[0] + ".wav")
            extract_audio_from_video(file, audio_output_path)

def get_feature(file_path: str, feature_type:str="MFCC", mean_signal_length:int=110000, embed_len: int = 39):
    feature = None
    signal, fs = librosa.load(file_path)# Default setting on sampling rate
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    if feature_type == "MFCC":
        mfcc =  librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len)
        feature = np.transpose(mfcc)
    return np.array([feature])

def dataset_from_directories(directories, classes):
    dataset = {}
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            modality, vc, emotion, intensity, statement, repeat, actor = map(int,os.path.basename(file)[:-4].split("-"))
            if modality == 2 or vc == 2:
                continue
            dataset[classes[emotion-1]] = dataset.get(classes[emotion-1],[]) + [os.path.join(directory,file)]
    return dataset

def load_combined_dataset(directory, class_names, img_size=(224, 224)):
    images = []
    audio_features = []
    labels = []
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        video_dir = os.path.join(class_dir, "video")
        audio_dir = os.path.join(class_dir, "audio")

        if os.path.isdir(video_dir) and os.path.isdir(audio_dir):
            for video_folder in os.listdir(video_dir):
                video_folder_path = os.path.join(video_dir, video_folder)
                audio_file_path = os.path.join(audio_dir, video_folder + ".wav")

                if os.path.isdir(video_folder_path) and os.path.isfile(audio_file_path):
                    # Process video frames
                    features = get_feature(audio_file_path)
                    for frame_file in os.listdir(video_folder_path):
                        if frame_file.endswith(".jpg"):
                            img_path = os.path.join(video_folder_path, frame_file)
                            img = cv2.imread(img_path)
                            img = cv2.resize(img, img_size)
                            images.append(img)
                            labels.append(class_indices[class_name])
                            audio_features.append(features)

    images = np.array(images) / 255.0  # Normalize images
    audio_features = np.array(audio_features)
    labels = np.array(labels)

    return images, audio_features, labels

def cache_dataset(images, audio_features, labels, path):
    np.savez(path, images=images, audio_features=audio_features, labels=labels)

def load_cached_dataset(path):
    data = np.load(path)
    return data["images"], data["audio_features"], data["labels"]
import streamlit as st
import os
import argparse
import shutil
import cv2
import numpy as np
import whisper
from pydub import AudioSegment
from torch import cuda
from multinet.model.multinet import MultiNet
from moviepy.editor import VideoFileClip
from multinet.data.utils import get_random_frames, detect_and_crop_faces, get_feature

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
speech_model = whisper.load_model("tiny", device=device)

def segments_speech(video_file):
    transcription_result = speech_model.transcribe(video_file)
    temp_file = os.path.join(os.getcwd(), "temp_"+str(os.urandom(5).hex()))
    os.makedirs(temp_file)

    segments = []

    video = VideoFileClip(video_file)
    for i, segment in enumerate(transcription_result['segments']):
        start_time = segment['start']
        end_time = segment['end'] + 0.5
        
        # Load the video
        
        # Clip the video segment
        clipped_video = video.subclip(start_time, end_time)
        
        # Save the clipped video
        output_file = os.path.join(temp_file, f"segment_{i}.mp4")
        clipped_video.write_videofile(output_file)
        
        # Close the video clip to free up memory
        clipped_video.close()

        segments.append([output_file, segment['text']])

    video.close()

    return temp_file, segments
    

def process_video(video_path, img_size=(224, 224), num_frames=3):
    frames = get_random_frames(video_path, num_frames)
    processed_frames = []

    for frame in frames:
        face = detect_and_crop_faces(frame)
        if face is not None:
            face_resized = cv2.resize(face, img_size)
            processed_frames.append(face_resized)

    processed_frames = np.array(processed_frames) / 255.0  # Normalize the frames
    return processed_frames


def process_audio(video_path):
    return get_feature(video_path)


model = MultiNet(
    timnet_input_shape=(215, 39),  # Adjust input shapes as per your data
    class_labels=("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    p_att_lite_input_shape=(224, 224, 3),
    dropout_rate=0.5
)

model.compile_model(learning_rate=0.001)
model.combined_model.load_weights("models/model_epoch_99.h5")

model = model.combined_model


def run_inference(model, video_path):
    frames = process_video(video_path)
    audio_features = process_audio(video_path)
    audio_features = np.repeat(audio_features, 3, axis=0)

    predictions = model.predict([audio_features, frames])
    preds = np.argmax(predictions, axis=1)
    classes = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
    pred = np.bincount(preds).argmax()
    return classes[pred]

st.title("Video Emotion Recognition with MultiNet")
st.write("Upload a video file to get started")

video_file = st.file_uploader("Upload Video", type=["mp4"])

if video_file:
    with st.spinner('Processing...'):
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        temp_dir, segments = segments_speech("temp_video.mp4")
        results = []
        for i, (segment_video_path, text) in enumerate(segments):
            predictions = run_inference(model, segment_video_path)
            results.append({"Segment": f"Segment {i+1}", "Transcript": text, "Emotion": predictions})
        shutil.rmtree(temp_dir)
        # Delete temp_video.mp4 file
        os.remove("temp_video.mp4")
    
    st.write("### Results")
    st.table(results)

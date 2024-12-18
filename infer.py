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

def segments_speech(video_file):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    speech_model = whisper.load_model("tiny", device=device)
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
    """
    Process the video to extract random frames, resize them, and detect faces.
    """
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
    """
    Extract audio from video and convert it into features.
    """
    # Extract audio feature directly from the video file path using librosa
    return get_feature(video_path)


def load_multinet_model(weights_path):
    """
    Load the pre-trained MultiNet model from weights.
    """
    model = MultiNet(
        timnet_input_shape=(215, 39),  # Adjust input shapes as per your data
        class_labels=("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
        p_att_lite_input_shape=(224, 224, 3),
        dropout_rate=0.5
    )
    
    # Compile the model (with the same configuration as used during training)
    model.compile_model(learning_rate=0.001)
    
    # Load the trained model weights
    model.combined_model.load_weights(weights_path)

    return model.combined_model


def run_inference(model, video_path):
    """
    Run inference on the video file using the pre-trained MultiNet model.
    """
    # Process video frames
    frames = process_video(video_path)

    # Process audio
    audio_features = process_audio(video_path)
    audio_features = np.repeat(audio_features, 3, axis=0)
    print(audio_features.shape)
    print(frames.shape)

    # Make prediction using the combined video frames and audio features
    predictions = model.predict([audio_features, frames])
    preds = np.argmax(predictions, axis=1)
    classes = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
    pred = np.bincount(preds).argmax()
    return classes[pred]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a video using the MultiNet model.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the pre-trained model weights.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    
    args = parser.parse_args()

    weights_path = args.weights_path
    video_path = args.video_path

    model = load_multinet_model(weights_path)
    temp_dir, segments = segments_speech(video_path)
    # Load the model

    # # Run inference on the video
    for i, (segment_video_path, text) in enumerate(segments):
        predictions = run_inference(model, segment_video_path)
        print(f"Segment {i+1} => Transcript: {text} | Emotion: {predictions}")

    shutil.rmtree(temp_dir)


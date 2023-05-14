import os
from os import listdir
import librosa
import soundfile
import os, glob, pickle
import numpy as np
def remove_files():
    current_working_directory = os.getcwd()
    files=os.listdir(current_working_directory)
    filtered_files=[file for file in files if file.endswith(".wav")]
    for file in filtered_files:
        path_to_file = os.path.join(current_working_directory, file)
        os.remove(path_to_file)
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def predict_output(filename,modelname):
    feature=extract_feature(filename, mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)
    loaded_model=pickle.load(open(modelname, 'rb'))
    prediction=loaded_model.predict(feature)
    return prediction
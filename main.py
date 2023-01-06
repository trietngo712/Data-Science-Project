import os
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import noisereduce

#convert every audio into a numerical vector with the same dimension e.g, common set of features
def convertAudioToFeature(pathToAudio, numberOfSecond):
    rate, audio = wf.read(pathToAudio)

    #cancel white noise
    audio = noisereduce.reduce_noise(y=audio, sr=rate)

    audioLength = audio.shape[0] / rate

    if audioLength > numberOfSecond:
        #if the audio is longer a defined number of seconds
        #its length is cut so that it is as long as the define number of seconds
        audio = audio[0:rate * numberOfSecond]
    else:
        #if the audio is shorter than a defined number of seconds 
        #the audio vector is padded with zeroes
        audio = np.pad(audio, int(rate*numberOfSecond - len(audio)), 'constant')
    
    #Mel-frequency cepstral coefficients
    #Representation of the short-term power spectrum of a sound
    mfcc = librosa.feature.mfcc(y = audio.astype(float), sr = rate)

    return mfcc


def read(source):

    df = pd.read_csv(source)

    X_mfcc = []
    for audioPath in df['path']:
        mfcc = convertAudioToFeature(audioPath, 2)
        X_mfcc.append(mfcc)
    




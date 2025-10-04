import torch
from torch import nn
import tensorflow as tf

# encoder receives an input (series of text tokens) and builds a representation of its features
# trained to acquire an understanding

# decoder uses those representations alongside other inputs to generate a sequence
# trained to generate outputs

# now need audio

# mp3 -> WAV -> Spectrogram  -> pitch detection -> note grouping -> tokenization -> transformer model -> prediction

# ffmpeg or py- audio decoding
# numpy - feature extraction
# deep learning - torch
# transformer weights - transformers
# visualization - matplotlib

# MP3 -> waveform
# pytorch, torchaudio, librosa?, 
# pyaudio captures live audio from a microphone

# pyaudio -> torchaudio -> feature extraction? -> transformer -> flask API w/ endpoint
# madmom apparently has pre-trained models for musical genres

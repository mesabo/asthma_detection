#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, IIST
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""


# Real-time microphone capture

import pyaudio
import wave
import os

def record_audio(filename="recorded.wav", duration=5, sr=22050):
    """
    Record lung audio from microphone and save to a WAV file.

    Args:
        filename (str): Output filename (with full or relative path)
        duration (int): Recording duration in seconds
        sr (int): Sample rate (Hz)
    """
    channels = 1
    chunk = 1024
    format = pyaudio.paInt16

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=sr,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"🎙️ Recording {duration}s of audio...")
    frames = []

    for _ in range(0, int(sr / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("✅ Recording complete.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()
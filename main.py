import sounddevice as sd
from scipy.io import wavfile


def record():
    fs = 16000  # Sample rate
    seconds = 8  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    return fs, myrecording


if __name__ == '__main__':
    print("Type enter when you will be ready to record an audio: ")
    input()
    print("Recording...")
    sample_rate, signal = record()
    wavfile.write("sample.wav", sample_rate, signal)
    print("Done!")

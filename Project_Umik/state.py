# state.py

fft_spectrum = None

def set_fft(data):
    global fft_spectrum
    fft_spectrum = data

def get_fft():
    return fft_spectrum

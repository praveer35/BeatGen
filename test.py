import pretty_midi
import numpy as np
from scipy.io.wavfile import write

def midi_to_wav(midi_file, output_wav_file):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Synthesize the MIDI data to get audio as a NumPy array
    audio_data = midi_data.synthesize()
    
    # Define the sample rate (usually 44100 Hz)
    sample_rate = 44100
    
    # Write the audio data to a WAV file
    # Ensure audio_data is in 16-bit format for WAV files
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    write(output_wav_file, sample_rate, audio_data)
    print(f"Converted {midi_file} to {output_wav_file}")

if __name__ == "__main__":
    midi_file = 'output.mid'
    output_wav_file = 'output.wav'
    
    midi_to_wav(midi_file, output_wav_file)

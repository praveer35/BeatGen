from midiutil import MIDIFile

# Create a new MIDI file with 2 tracks (melody + chords)
midi = MIDIFile(2)
track_melody = 0
track_chords = 1
time = 0  # Start at the beginning
tempo = 82  # BPM
midi.addTempo(track_melody, time, tempo)
midi.addTempo(track_chords, time, tempo)

# Define a more volatile melody (note, start time in beats, duration in beats)
melody = [
    (58, 0, 0.5),  # A#3
    (63, 0.5, 0.25),  # Eb4 (leap up)
    (61, 0.75, 0.25),  # Db4 (chromatic down)
    (65, 1, 0.75),  # F4 (leap up)
    (60, 1.75, 0.25),  # C4 (drop)
    
    (67, 2, 0.5),  # G4
    (70, 2.5, 0.5),  # A#4 (big leap up)
    (66, 3, 0.5),  # F#4 (tense chromatic descent)
    (63, 3.5, 1),  # Eb4 (resolution)
    
    (62, 4.5, 0.25),  # D4
    (65, 4.75, 0.25),  # F4
    (60, 5, 0.5),  # C4 (drop)
    (67, 5.5, 0.5),  # G4
    (63, 6, 0.5),  # Eb4
    (71, 6.5, 0.75),  # B4 (huge leap, unstable)
    (65, 7.25, 0.75),  # F4 (back down)
]

# Define the chord progression (chord notes, start time in beats, duration in beats)
chords = [
    ([48, 51, 55, 58], 0, 2),  # Cm (C-Eb-G-Bb)
    ([43, 47, 50, 53], 2, 2),  # G7 (G-B-D-F)
    ([46, 50, 53, 56], 4, 2),  # Bb7 (Bb-D-F-Ab)
    ([44, 48, 51, 55], 6, 2),  # Abmaj7 (Ab-C-Eb-G)
]

# Add melody notes to MIDI file
for note, start, duration in melody:
    midi.addNote(track_melody, channel=0, pitch=note, time=start, duration=duration, volume=100)

# Add chords to MIDI file
for chord, start, duration in chords:
    for note in chord:
        midi.addNote(track_chords, channel=1, pitch=note, time=start, duration=duration, volume=80)

# Save the MIDI file
with open("dark_hiphop_volatile.mid", "wb") as midi_file:
    midi.writeFile(midi_file)

print("MIDI file saved as 'dark_hiphop_volatile.mid'")

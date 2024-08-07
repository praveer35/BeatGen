import asyncio
import fluidsynth

# Define your tracks as lists of tuples: (note, start_time, end_time)
tracks = [
    [(60, 0, 4), (64, 0, 4), (67, 0, 4)],  # Track 1
    [(72, 0, 4)],  # Track 2
    [(70, 0, 1), (72, 0.5, 1.5), (74, 1, 2.5)]  # Track 3
]


# Initialize the synthesizer
soundfont = "Yamaha_C3_Grand_Piano.sf2"  # Replace with the path to your SoundFont file
synth = fluidsynth.Synth()
synth.start()

# Load the SoundFont
sfid = synth.sfload(soundfont)

# Select program for each channel
for channel in range(len(tracks)):
    synth.program_select(channel, sfid, 0, 0)

async def play_note_on_channel(synth, note, start_time, end_time, channel):
    await asyncio.sleep(start_time)
    synth.noteon(channel, note, 100)
    await asyncio.sleep(end_time - start_time)
    synth.noteoff(channel, note)

async def play_track(synth, track, channel):
    tasks = [play_note_on_channel(synth, note, start_time, end_time, channel) for note, start_time, end_time in track]
    await asyncio.gather(*tasks)

async def main_player():
    tasks = [play_track(synth, track, channel) for channel, track in enumerate(tracks)]
    await asyncio.gather(*tasks)
    synth.delete()

# Run the main function
asyncio.run(main_player())
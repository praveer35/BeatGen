import fluidsynth
import asyncio
import threading
import lib

class PlaybackManager:
    """ Manages playback, allowing single-play or looping mode. """
    def __init__(self):
        self.loop = None
        self.stop_event = None
        self.synth = None
        self.thread = None
        self.is_playing = False  # Tracks if playback is ongoing

    def start_playback(self, measures, velocities, bpm, transpose, soundfont_map, loop=True):
        """ Starts playback with an option to loop or play once. """
        if self.is_playing:
            return {"data": "playback already in progress"}  # Prevent starting if already playing

        self.stop_playback()  # Ensure any previous playback is fully stopped

        self.stop_event = asyncio.Event()
        self.loop = asyncio.new_event_loop()  # Always create a fresh event loop
        self.thread = threading.Thread(
            target=self._run_loop, 
            args=(measures, velocities, bpm, transpose, soundfont_map, loop),
            daemon=True
        )
        self.thread.start()
        return {"data": "playback started"}

    def _run_loop(self, measures, velocities, bpm, transpose, soundfont_map, loop):
        """ Runs playback asynchronously in a separate thread. """
        asyncio.set_event_loop(self.loop)
        
        self.synth = fluidsynth.Synth()
        self.synth.start()

        channel_velocities = []
        tracks = []
        sfids = []

        for key in measures.keys():
            if key == 'drums':
                continue
            synth_track = lib.synth_convert(measures[key], transpose)
            channel_velocities.append(velocities[key])
            tracks.append(synth_track)
            sfids.append(self.synth.sfload(f'Soundfonts/{soundfont_map[key]}.sf2'))

        for channel in range(len(tracks)):
            self.synth.program_select(channel, sfids[channel], 0, 0)

        async def play_note_on_channel(synth, note, start_time, end_time, channel):
            await asyncio.sleep(start_time * 60 / bpm)
            synth.noteon(channel, note, channel_velocities[channel])
            await asyncio.sleep((end_time - start_time) * 60 / bpm)
            synth.noteoff(channel, note)

        async def play_track(synth, track, channel):
            tasks = [play_note_on_channel(synth, note, start_time, end_time, channel) for note, start_time, end_time in track]
            await asyncio.gather(*tasks)

        async def main_player():
            """ Handles both looping and single-play mode. """
            self.is_playing = True  # Mark as playing
            while not self.stop_event.is_set():
                tasks = [play_track(self.synth, track, channel) for channel, track in enumerate(tracks)]
                await asyncio.gather(*tasks)
                if not loop:  # Stop after one playthrough if loop=False
                    break
            self.is_playing = False  # Mark as not playing when done
            self.synth.delete()

        self.loop.create_task(main_player())
        self.loop.run_forever()

    def stop_playback(self):
        """ Stops playback, ensuring complete cleanup before restarting. """
        if self.stop_event:
            self.stop_event.set()

        if self.is_playing:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)  
            if self.synth:
                self.synth.delete()
                self.synth = None  # Ensure old fluidsynth instance is removed
            if self.thread:
                self.thread.join()  # Ensure full cleanup before restarting
                self.thread = None  # Avoid reusing the old thread
            self.loop = None  # Ensure a fresh event loop on next start
            self.is_playing = False  # Reset the play state

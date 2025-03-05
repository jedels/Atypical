import sounddevice as sd
import numpy as np
import pyttsx3  # For TTS
import threading
import datetime
import scipy.io.wavfile as wav
from scipy import signal as sig
from typing import Any, List, Optional, Tuple
from speechbrain.inference.ASR import StreamingASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
import torch
import time
import pyaudio
import queue
import signal
import sys
import os
import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SR = 16000

class SimpleVoiceCloner:
    def __init__(self):
        # Find appropriate output device
        target_id = -1
        for i, device in enumerate(sd.query_devices()):
            if device['max_output_channels'] > 0 and "OpenMove" in device['name']:
                target_id = i
                break
        
        # Fallback to default device if OpenMove not found
        if target_id < 0:
            for i, device in enumerate(sd.query_devices()):
                if device['max_output_channels'] > 0:
                    target_id = i
                    print(f"Using default output device: {device['name']}")
                    break
        
        self.playing = False
        self._play_thread = None
        self.device_id = target_id
        
        # Store recorded audio for playback
        self.recorded_audio = None
        self.recorded_sample_rate = 16000
        self.voice_sample_path = "voice_sample.wav"
        
        # Voice characteristics
        self.pitch_shift = 1.0
        self.tempo = 1.0
        self.voice_samples = []
        self.max_samples = 5
        
        # Cloning buffer
        self.cloning_buffer = []
        self.max_cloning_buffer = 5  # Keep last 5 seconds of audio for cloning
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
    
    def update_voice_sample(self, audio_data, sample_rate=16000, transcript=None):
        """Save audio for playback and analysis"""
        try:
            if len(audio_data) > 0:
                # Store for playback
                self.recorded_audio = audio_data.copy()
                self.recorded_sample_rate = sample_rate
                
                # Add to voice samples for analysis
                self.voice_samples.append(audio_data.copy())
                if len(self.voice_samples) > self.max_samples:
                    self.voice_samples.pop(0)
                
                # Extract voice characteristics
                self._analyze_voice(audio_data, sample_rate)
                
                # Save as WAV file
                normalized_audio = audio_data / np.max(np.abs(audio_data))
                wav.write(self.voice_sample_path, sample_rate, normalized_audio.astype(np.float32))
                
                print(f"Stored {len(audio_data)/sample_rate:.2f}s of audio for voice cloning")
                return True
        except Exception as e:
            print(f"Error saving audio: {e}")
        return False
    
    def _analyze_voice(self, audio_data, sample_rate):
        """Extract basic voice characteristics"""
        try:
            # Simple pitch detection via zero crossings
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            if len(zero_crossings) > 10:
                avg_samples_between = len(audio_data) / len(zero_crossings)
                freq = sample_rate / (2 * avg_samples_between)
                # Rough pitch estimation (very simplified)
                if 50 < freq < 500:  # Typical human voice range
                    # Calculate shift from average speaking pitch
                    avg_pitch = 120  # Average speaking pitch in Hz
                    # Adjust to create a target pitch shift
                    self.pitch_shift = avg_pitch / freq
                    print(f"Estimated voice pitch: {freq:.1f}Hz, shift factor: {self.pitch_shift:.2f}")
            
            # Simple tempo detection via amplitude envelope
            envelope = np.abs(sig.hilbert(audio_data))
            # Find peaks in envelope for rhythm
            peaks, _ = sig.find_peaks(envelope, height=0.1, distance=sample_rate/10)
            if len(peaks) > 2:
                avg_peak_distance = np.mean(np.diff(peaks))
                # Normalize to typical speech tempo
                self.tempo = 0.2 / (avg_peak_distance / sample_rate)
                self.tempo = max(0.8, min(1.2, self.tempo))  # Limit range
                print(f"Estimated tempo factor: {self.tempo:.2f}")
                
        except Exception as e:
            print(f"Error analyzing voice: {e}")
    
    def play_recorded_voice(self):
        """Play back the recorded voice sample"""
        if self.recorded_audio is not None and len(self.recorded_audio) > 0:
            print("Playing back recorded voice...")
            print(f"Recorded audio length: {len(self.recorded_audio)} samples")  # Debug statement
            sd.play(self.recorded_audio, self.recorded_sample_rate, device=self.device_id)
            sd.wait()
            return True
        else:
            print("No recorded voice available")
            print("Recorded audio is None or empty.")  # Debug statement
            return False
    
    def play(self, text: str, sample_rate: int = 16000):
        """
        Play synthesized speech in a non-blocking manner
        """
        if self.playing:
            self.stop()
            
        self.playing = True
        self._play_thread = threading.Thread(
            target=self._play_audio,
            args=(text, sample_rate)
        )
        self._play_thread.start()
        
        return self._play_thread
    
    def _play_audio(self, text: str, sample_rate: int = 16000):
        try:
            print(f"Synthesizing speech for: '{text}'")
            
            # Adjust TTS engine properties based on analyzed voice
            self.tts_engine.setProperty('rate', int(self.tts_engine.getProperty('rate') * self.tempo))
            self.tts_engine.setProperty('pitch', int(self.tts_engine.getProperty('pitch') * self.pitch_shift))
            
            # Generate and play TTS audio
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
                
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
        finally:
            self.playing = False
    
    def stop(self):
        """Stop any currently playing audio"""
        if self.playing:
            sd.stop()
            if self._play_thread:
                self._play_thread.join()
            self.playing = False

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self.playing

    def save_cloned_sample(self, audio_data, sample_rate=16000):
        """Save the cloned voice sample as a .wav file with a timestamp"""
        try:
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cloned_sample_{timestamp}.wav"
            
            # Normalize audio data
            normalized_audio = audio_data / np.max(np.abs(audio_data))
            # Save as WAV file
            wav.write(filename, sample_rate, normalized_audio.astype(np.float32))
            print(f"Cloned sample saved as {filename}")
        except Exception as e:
            print(f"Error saving cloned sample: {e}")

    def update_cloning_buffer(self, audio_data, sample_rate=16000):
        """Update the buffer used for voice cloning"""
        self.cloning_buffer.append(audio_data)
        buffer_duration = sum(len(chunk) for chunk in self.cloning_buffer) / sample_rate
        while buffer_duration > 5.0:  # Keep only last 5 seconds
            self.cloning_buffer.pop(0)
            buffer_duration = sum(len(chunk) for chunk in self.cloning_buffer) / sample_rate

    def clone_voice_with_audio(self, text, audio_data, sample_rate=16000):
        """Clone voice using both text and audio"""
        # Update voice characteristics based on the audio
        self.update_voice_sample(audio_data, sample_rate, text)
    
        # Adjust TTS engine properties based on analyzed voice
        try:
            current_rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', int(current_rate * self.tempo))
        except Exception as e:
            print(f"Warning: Couldn't adjust speech rate: {e}")

        try:
            current_pitch = self.tts_engine.getProperty('pitch')
            if current_pitch is not None:
                self.tts_engine.setProperty('pitch', int(current_pitch * self.pitch_shift))
            else:
                print("Warning: Pitch adjustment not supported on this system")
        except Exception as e:
            print(f"Warning: Couldn't adjust pitch: {e}")
    
        # Generate and play TTS audio
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error in TTS playback: {e}")
            return False

    def save_cloned_voice_transcription(self, text, filename):
        """Save the cloned voice transcription"""
        try:
            with open(filename, 'w') as f:
                f.write(text)
            print(f"Cloned voice transcription saved to {filename}")
        except Exception as e:
            print(f"Error saving cloned voice transcription: {e}")

class speech_pipeline(object):
    def __init__(self, chunk_cfg=None, SR=16000, checkpoint_path=None):
        min_chunk_size = chunk_cfg.chunk_size * 639 / SR
        self.min_chunk_size = min_chunk_size
        self.chunk_samples = int(min_chunk_size * SR)
    
        # Set up ASR model
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading ASR model from checkpoint: {checkpoint_path}")
                self.asr_model = StreamingASR.from_hparams(
                    source = "speechbrain/asr-streaming-conformer-gigaspeech",
                    savedir = "./checkpoints",
                    run_opts={"device":"cpu"},
                    hparams_file=checkpoint_path
                )
            else:
                print("Using default ASR model")
                self.asr_model = StreamingASR.from_hparams(
                    source = "speechbrain/asr-streaming-conformer-gigaspeech",
                    savedir = "./checkpoints",
                    run_opts={"device":"cpu"} 
                )
            
            self.context = self.asr_model.make_streaming_context(chunk_cfg)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            self.model_loaded = False
        
        self.pred = ""
        self.index = 0
        self.output_silence = False  # Flag to control whether to output silence markers
    
    def stream_audio(self, chunk):
        if not self.model_loaded:
            return ""
            
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk)

        rel_length = torch.tensor([1.0])
        if chunk.shape[-1] < self.chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, self.chunk_samples - chunk.shape[-1]))
            
        try:
            words = self.asr_chunk(self.context, chunk, rel_length)
            curr_words = ""
            for w in words:
                curr_words += w 

            curr_words = curr_words.lower()

            # Only handle silence if we want to output it
            if curr_words == "" and self.index > 0:
                if self.output_silence:
                    curr_words = " |SILENCE >"
                else:
                    curr_words = ""
            
            # Only add to prediction and print if there's actual content
            if curr_words and not curr_words.isspace():
                self.pred += curr_words
                # Only log non-empty output
                if not curr_words.startswith(" |SILENCE"):
                    print(f"ASR at {self.index * self.min_chunk_size:.2f}s: {curr_words}")
            
            self.index = self.index + 1

            return curr_words
        except Exception as e:
            print(f"Error in ASR: {e}")
            return ""
            
    def asr_chunk(self, context, chunk, chunk_len):
        if chunk_len is None:
            chunk_len = torch.ones((chunk.size(0),))

        chunk = chunk.float()
        chunk, chunk_len = chunk.to(self.asr_model.device), chunk_len.to(self.asr_model.device)
        t0 = time.time()
        x = self.asr_model.encode_chunk(context, chunk, chunk_len)
        words, _ = self.asr_model.decode_chunk(context, x)
        print(f"ASR runtime: {time.time() - t0:.2f}s")
        return words


class AudioProcessorApp:
    def __init__(self, master, speechpipeline, voice_cloner, sample_rate=16000, channels=1, save_folder=None):
        self.master = master
        self.master.title("Speech Recognition with Voice Cloning")
        self.master.geometry("800x600")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.speechpipeline = speechpipeline
        self.voice_cloner = voice_cloner
        self.sample_rate = sample_rate
        self.channels = channels
        self.save_folder = save_folder
        
        # Audio settings
        self.chunk_size = speechpipeline.chunk_samples
        
        # Thread-safe queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Keep a buffer of raw audio for playback
        self.raw_audio_buffer = []
        self.max_buffer_chunks = 50
        
        # Buffer of last words for context
        self.last_words = []
        self.max_words = 10
        
        # Threading control
        self.is_running = threading.Event()
        self.process_thread = None
        self.playback_thread = None
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.text_log = []
        
        # Auto-playback settings
        self.auto_playback = False
        self.playback_queue = queue.Queue()
        
        # Cloning text buffer
        self.cloning_text_buffer = []
        self.max_cloning_text_buffer = 10  # Keep last 10 words for cloning
        
        # Create GUI components
        self.setup_gui()
        
        # Start audio processing
        self.start()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Stop", command=self.toggle_processing)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Play recorded voice button
        self.play_btn = ttk.Button(control_frame, text="Play Recording", command=self.play_recording)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Clone voice button
        self.clone_btn = ttk.Button(control_frame, text="Clone Voice", command=self.clone_voice)
        self.clone_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-playback checkbox
        self.auto_var = tk.BooleanVar(value=self.auto_playback)
        self.auto_check = ttk.Checkbutton(
            control_frame, 
            text="Auto Voice Clone",
            variable=self.auto_var,
            command=self.toggle_auto_playback
        )
        self.auto_check.pack(side=tk.LEFT, padx=15)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Transcription tab
        transcription_frame = ttk.Frame(notebook)
        notebook.add(transcription_frame, text="Transcription")
        
        # Text area for transcriptions
        self.text_area = scrolledtext.ScrolledText(transcription_frame, wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Audio visualization tab
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio")
        
        # Audio visualization
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Audio Waveform")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-1, 1)
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=audio_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Voice characteristics tab
        voice_frame = ttk.Frame(notebook)
        notebook.add(voice_frame, text="Voice")
        
        # Voice characteristics display
        info_frame = ttk.LabelFrame(voice_frame, text="Voice Characteristics")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Pitch
        ttk.Label(info_frame, text="Estimated Pitch:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pitch_var = tk.StringVar(value="Not analyzed")
        ttk.Label(info_frame, textvariable=self.pitch_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tempo
        ttk.Label(info_frame, text="Speech Tempo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tempo_var = tk.StringVar(value="Not analyzed")
        ttk.Label(info_frame, textvariable=self.tempo_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Samples
        ttk.Label(info_frame, text="Voice Samples:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.samples_var = tk.StringVar(value="0 samples")
        ttk.Label(info_frame, textvariable=self.samples_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Custom text input for voice cloning
        input_frame = ttk.LabelFrame(voice_frame, text="Custom Voice Clone")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.custom_text = ttk.Entry(input_frame)
        self.custom_text.pack(fill=tk.X, padx=5, pady=5)
        
        clone_btn = ttk.Button(input_frame, text="Clone Voice for Text", command=self.clone_custom_text)
        clone_btn.pack(padx=5, pady=5)
        
        # Bottom status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_var = tk.StringVar(value="Ready to start")
        ttk.Label(status_frame, textvariable=self.info_var).pack(side=tk.LEFT)
    
    def update_gui(self):
        """Update GUI elements"""
        # Update voice characteristics display
        if hasattr(self.voice_cloner, 'pitch_shift') and self.voice_cloner.pitch_shift != 0:
            estimated_pitch = 120 / self.voice_cloner.pitch_shift if self.voice_cloner.pitch_shift != 0 else 0
            self.pitch_var.set(f"{estimated_pitch:.1f} Hz")
        else:
            self.pitch_var.set("Not analyzed")
            
        if hasattr(self.voice_cloner, 'tempo'):
            self.tempo_var.set(f"{self.voice_cloner.tempo:.2f}x")
        else:
            self.tempo_var.set("Not analyzed")
            
        if hasattr(self.voice_cloner, 'voice_samples'):
            samples_duration = sum(len(s) for s in self.voice_cloner.voice_samples) / self.sample_rate
            self.samples_var.set(f"{len(self.voice_cloner.voice_samples)} samples ({samples_duration:.1f}s)")
        else:
            self.samples_var.set("0 samples")
        
        # Update audio visualization if we have data
        if len(self.raw_audio_buffer) > 0:
            audio = np.concatenate(self.raw_audio_buffer[-5:])
            # Subsample for performance
            step = max(1, len(audio) // 1000)
            audio = audio[::step]
            
            # Update plot
            x = np.arange(len(audio))
            self.line.set_data(x, audio)
            self.ax.set_xlim(0, len(audio))
            self.fig.canvas.draw_idle()
        
        # Schedule next update if running
        if self.is_running.is_set():
            self.master.after(500, self.update_gui)
    
    def toggle_processing(self):
        """Toggle audio processing on/off"""
        if self.is_running.is_set():
            self.stop()
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Stopped")
        else:
            self.start()
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Running")
    
    def play_recording(self):
        """Play recorded voice"""
        if self.voice_cloner:
            self.status_var.set("Playing recording...")
            threading.Thread(target=self._play_recording_thread).start()
    
    def _play_recording_thread(self):
        """Thread for playing recording"""
        if self.voice_cloner.play_recorded_voice():
            self.master.after(0, lambda: self.status_var.set("Playback complete"))
        else:
            self.master.after(0, lambda: self.status_var.set("No recording available"))
        
        # Reset status after a delay
        self.master.after(2000, lambda: self.status_var.set("Ready"))
    
    def clone_voice(self):
        """Clone voice for last transcription"""
        if not self.cloning_text_buffer:
            self.status_var.set("No text to clone")
            return
            
        text = " ".join(self.cloning_text_buffer)
        audio = np.concatenate(self.voice_cloner.cloning_buffer)
        self.status_var.set(f"Cloning voice for: {text}")
        threading.Thread(target=self._clone_voice_thread, args=(text, audio)).start()
    
    def clone_custom_text(self):
        """Clone voice for custom text"""
        text = self.custom_text.get().strip()
        if not text:
            self.status_var.set("Enter text to clone")
            return
            
        self.status_var.set(f"Cloning voice for: {text}")
        audio = np.concatenate(self.voice_cloner.cloning_buffer) if self.voice_cloner.cloning_buffer else None
        threading.Thread(target=self._clone_voice_thread, args=(text, audio)).start()
    
    def _clone_voice_thread(self, text, audio):
        if self.voice_cloner:
            success = self.voice_cloner.clone_voice_with_audio(text, audio, self.sample_rate)
            if success:
                self.master.after(0, lambda: self.status_var.set("Playback complete"))
                
                # Save the cloned voice transcription
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                transcription_filename = os.path.join(self.save_folder, f"cloned_voice_transcription_{timestamp}.txt")
                self.voice_cloner.save_cloned_voice_transcription(text, transcription_filename)
                
                # Save the cloned audio sample
                if self.voice_cloner.recorded_audio is not None:
                    self.voice_cloner.save_cloned_sample(self.voice_cloner.recorded_audio, self.voice_cloner.recorded_sample_rate)
            else:
                self.master.after(0, lambda: self.status_var.set("Cloning failed"))
        else:
            self.master.after(0, lambda: self.status_var.set("Voice cloner not available"))
        
        self.master.after(2000, lambda: self.status_var.set("Ready"))
    
    def toggle_auto_playback(self):
        """Toggle auto voice cloning"""
        self.auto_playback = self.auto_var.get()
        self.info_var.set(f"Auto voice clone: {'ON' if self.auto_playback else 'OFF'}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio to handle incoming audio data"""
        if self.is_running.is_set():
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to raw audio buffer for playback
            self.raw_audio_buffer.append(audio_data.copy())
            if len(self.raw_audio_buffer) > self.max_buffer_chunks:
                self.raw_audio_buffer.pop(0)  # Remove oldest chunk
                
            # Add to processing queue
            self.audio_queue.put(audio_data)
            
        return (in_data, pyaudio.paContinue)
    
    def update_text(self, text):
        """Update the text display with new transcription"""
        # Append text to the text area
        self.text_area.insert(tk.END, text + " ")
        self.text_area.see(tk.END)  # Scroll to the end
        
        # Update the last words display
        last_words = " ".join(self.last_words)
        self.info_var.set(f"Last words: {last_words}")
    
    def playback_worker(self):
        """Thread function to handle auto-playback of voice cloned text"""
        while self.is_running.is_set():
            try:
                text, audio = self.playback_queue.get(timeout=0.5)
                
                if self.voice_cloner and not self.voice_cloner.is_playing():
                    time.sleep(0.5)
                    print(f"Auto voice clone playback: '{text}'")
                    
                    self.master.after(0, lambda: self.status_var.set(f"Auto-cloning: {text}"))
                    
                    # Clone voice using both text and audio
                    success = self.voice_cloner.clone_voice_with_audio(text, audio, self.sample_rate)
                    
                    if success:
                        # Save the cloned voice transcription
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        transcription_filename = os.path.join(self.save_folder, f"auto_cloned_voice_transcription_{timestamp}.txt")
                        self.voice_cloner.save_cloned_voice_transcription(text, transcription_filename)
                    
                    self.master.after(3000, lambda: self.status_var.set("Ready"))
                
                self.playback_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in playback worker: {e}")
                self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
    
    def process_audio(self):
        """Thread function to process audio chunks"""
        while self.is_running.is_set():
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_chunk = audio_chunk[np.newaxis, :]
                
                # Process with ASR
                current_words = self.speechpipeline.stream_audio(audio_chunk)
                self.audio_queue.task_done()
                
                # Only process non-empty words
                if current_words and current_words.strip() and not current_words.startswith(" |SILENCE"):
                    # Filter out silence markers
                    cleaned_words = current_words.replace("|SILENCE", "").replace(">", "").strip()
                    
                    if cleaned_words:
                        # Only log non-empty output
                        print(f"Recognized: '{cleaned_words}'")
                        self.text_log.append(cleaned_words)
                        
                        # Update GUI with new text
                        self.master.after(0, lambda text=cleaned_words: self.update_text(text))
                        
                        # Update cloning buffers
                        self.voice_cloner.update_cloning_buffer(audio_chunk[0], self.sample_rate)
                        self.cloning_text_buffer.extend(cleaned_words.split())
                        self.cloning_text_buffer = self.cloning_text_buffer[-self.max_cloning_text_buffer:]
                        
                        # Add this segment to playback queue if auto-playback is on
                        if self.auto_playback and self.voice_cloner:
                            self.playback_queue.put((cleaned_words, audio_chunk[0]))
                        
                        # Update word buffer
                        words = cleaned_words.strip().split()
                        for word in words:
                            if word and word != "|SILENCE" and word != ">":
                                self.last_words.append(word)
                        
                        # Keep only the last N words
                        self.last_words = self.last_words[-self.max_words:]
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
    
    def start(self):
        """Start audio processing"""
        self.is_running.set()
        
        # Clear text area
        self.text_area.delete(1.0, tk.END)
        
        # Set up audio stream 
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_worker)
        self.playback_thread.start()
        
        # Start GUI updates
        self.update_gui()
        
        # Update status
        self.status_var.set("Running")
        self.info_var.set("Listening...")
        
        print("Audio processing started with voice cloning.")
    
    def stop(self):
        """Stop audio processing gracefully"""
        print("\nStopping audio processing...")
        self.is_running.clear()
            
        # Save log if needed
        if self.save_folder is not None:
            try:
                with open(os.path.join(self.save_folder, "ASR_log.txt"), "w") as file:
                    for item in self.text_log:
                        file.write(item + "\n")
                print(f"Log saved to {os.path.join(self.save_folder, 'ASR_log.txt')}")
            except Exception as e:
                print(f"Error saving log: {e}")
    
        # Clean up audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Stop voice cloner if playing
        if self.voice_cloner and self.voice_cloner.is_playing():
            self.voice_cloner.stop()
        
        # Update status
        self.status_var.set("Stopped")
        self.info_var.set("Processing stopped")
        
        print("Audio processing stopped cleanly.")
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_running.is_set():
            self.stop()
            
        # Close audio
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
            
        # Wait for threads to finish (non-blocking)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
            
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
            
        self.master.destroy()
        
    def get_last_words(self):
        """Return the last few words as a string"""
        return " ".join(self.last_words)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-u", "--user", default="default_user", help="username")
    parser.add_argument("-m", "--section", default="default_section", help="section name")
    parser.add_argument("-c", "--checkpoint", default="CKPT+2025-01-28+16-39-43+00", help="model checkpoint to use")
    parser.add_argument("-s", "--show-silence", action="store_true", help="show silence markers in output")
    parser.add_argument("-a", "--auto-playback", action="store_true", help="enable auto voice clone playback")
    
    # Read arguments from command line
    args = parser.parse_args()
    user_name = args.user
    folder_name = "./material/" + args.section 
    user_folder = os.path.join("./results", user_name)
    os.makedirs(user_folder, exist_ok=True)
    save_folder = os.path.join(user_folder, args.section)
    os.makedirs(save_folder, exist_ok=True)
    
    # Checkpoint path
    checkpoint_path = os.path.join("./checkpoints", args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using default model.")
        checkpoint_path = None
    
    # Load memory if needed
    mem = ""
    mem_path = os.path.join(folder_name, "mem.txt")
    if os.path.exists(mem_path):
        with open(mem_path, "r") as file:
            mem = file.read()
    
    print("="*50)
    print("ASR with GUI and Voice Cloning")
    print("="*50)
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"User: {user_name}")
    print(f"Section: {args.section}")
    print("="*50)
    
    try:
        # Initialize voice cloner
        voice_cloner = SimpleVoiceCloner()
        
        # Set up speech pipeline with custom checkpoint
        step_size = 24
        context_len = 4
        chunk_cfg = DynChunkTrainConfig(step_size, context_len)
        
        pipeline0 = speech_pipeline(chunk_cfg, SR, checkpoint_path)
        # Set silence output based on command line arg
        pipeline0.output_silence = args.show_silence
        
        # Initialize Tkinter root
        root = tk.Tk()
        
        # Initialize the app
        app = AudioProcessorApp(
            master=root,
            speechpipeline=pipeline0,
            voice_cloner=voice_cloner,
            sample_rate=SR,
            save_folder=save_folder
        )
        
        # Set auto-playback from command line
        app.auto_playback = args.auto_playback
        app.auto_var.set(args.auto_playback)
        
        # Start the GUI main loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application closed")
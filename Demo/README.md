# Voice Cloning Demo

A real-time speech recognition and voice cloning application with GUI interface.

## Features

- Real-time speech recognition
- Voice cloning capabilities
- Audio visualization
- Recording and playback functionality
- Customizable voice characteristics
- Timestamped audio file saving

## Files
- `simple_gui_clone7.py`: Main application script
- `voice_sample.wav`: Sample voice file for testing
- `checkpoints/`: Directory containing model checkpoints (see Installation section for download instructions)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jedels/Atypical
cd Atypical/Demo
```

2. Download the checkpoint files:
   - Create a `checkpoints` directory in the Demo folder
   - Download the following files and place them in the `checkpoints` directory:
     - `lm.ckpt` (202.58 MB)
     - `tokenizer.ckpt` (247 KB)
   - The checkpoint files can be downloaded from: [Google Drive Link - TO BE ADDED]

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python simple_gui_clone7.py -u username -m section_name --checkpoint-dir checkpoints
```

### Options:
- `--device-name`: Specify audio device
- `--show-silence`: Show silence markers
- `--auto-playback`: Enable automatic voice cloning
- `--sample-rate`: Set custom sample rate (default: 16000)

### GUI Features:
- Start/Stop button for audio processing
- Play Recording button for playback
- Clone Voice button for voice cloning
- Auto Voice Clone toggle
- Real-time audio visualization
- Voice characteristics display

## Requirements

- Python 3.7 or higher
- Working microphone
- Audio output device

## Dependencies

- PyTorch
- SpeechBrain
- sounddevice
- PyAudio
- numpy
- matplotlib
- scipy 
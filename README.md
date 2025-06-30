# Soccer Player Object Detection & Tracking

A real time soccer player detection and tracking system using a finetuned YOLOv11 model and computer vision.


## Requirements

### System Requirements
- Python 3.8 or higher (built and tested on 3.13)

### Dependencies
The following packages are required (see `requirements.txt`):

- `ultralytics>=8.0.0` - YOLOv8 model framework
- `opencv-python>=4.8.0` - Computer vision library
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - PyTorch vision utilities
- `scikit-learn>=1.3.0` - Machine learning utilities
- `matplotlib>=3.7.0` - Plotting and visualization

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mavericksxx/soccer-player-object-detection.git
   cd soccer-player-object-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   Make sure you have the following files in your project directory:
   - `best.pt` - Fine tuned YOLOv11 model
   - `15sec_input_720p.mp4` - Input video file

## Usage

### Basic Tracking
```bash
# Run with default settings (displays real-time results)
python main.py

# Run without display (faster processing)
python main.py --no-display
```

### Custom Options
```bash
# Specify custom input and output files
python main.py --input your_video.mp4 --output tracked_output.mp4

# Use a different model
python main.py --model your_model.pt --input video.mp4
```

### Analysis
```bash
# Analyze generated tracking results
python main.py --analyze

# Performance evaluation with plots
python evaluate.py --plot
```

### Command Line Arguments
- `--input, -i`: Input video path (default: `15sec_input_720p.mp4`)
- `--model, -m`: YOLO model path (default: `best.pt`)
- `--output, -o`: Output video path (default: `output_tracked.mp4`)
- `--no-display`: Disable real-time display for faster processing
- `--analyze`: Analyze existing tracking results

## Output Files

The system generates several output files:
- `output_tracked.mp4` - Video with tracking annotations
- `tracking_results.json` - Detailed tracking data
- `evaluation_metrics.json` - Performance metrics
- `trajectories.png` - Player trajectory visualization
- `timeline.png` - Detection timeline

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Model file not found**: Ensure `best.pt` is in the project directory

3. **Video file not found**: Check that your input video file exists and the path is correct

4. **CUDA/GPU issues**: If you encounter GPU-related errors, the system will automatically fall back to CPU processing

### Performance Tips

- Use `--no-display` flag for faster processing when you don't need real-time visualization
- For large videos, consider processing in smaller segments
- Ensure sufficient disk space for output files

## Project Structure

```
soccer-player-object-detection/
├── main.py                 # Main application script
├── player_tracker.py       # Player tracking implementation
├── evaluate.py             # Performance evaluation
├── requirements.txt        # Python dependencies
├── best.pt                 # Pre trained YOLOv11 model
├── 15sec_input_720p.mp4    # Sample input video
├── README.md               # Documentation
└── methodology_report.md   # Summary report
```
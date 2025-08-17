# Using Original Video Clips for Visualization

This document explains how to use the modified JAAD/PIE training utilities to visualize bounding boxes on frames extracted from original video clips instead of white canvases.

## Changes Made

The following files have been modified to support loading frames from video clips:

1. **`lib/utils/jaadpie_train_utils_cvae.py`** - Modified the `test()` function to accept an `original_videos_dir` parameter and extract frames from video clips
2. **`configs/jaad/jaad.py`** - Added `--original_videos_dir` command line argument
3. **`configs/pie/pie.py`** - Added `--original_videos_dir` command line argument
4. **Training scripts** - Updated to pass the parameter to the test function
5. **Evaluation scripts** - Updated to pass the parameter to the test function

## Usage

### Training with Original Video Clips

When running training or evaluation, you can now specify the directory containing your original video clips:

```bash
# For JAAD dataset
python tools/jaad/train_cvae.py --original_videos_dir /path/to/your/video/clips

# For PIE dataset  
python tools/pie/train_cvae.py --original_videos_dir /path/to/your/video/clips

# For evaluation
python tools/jaad/eval_cvae.py --original_videos_dir /path/to/your/video/clips
python tools/pie/eval_cvae.py --original_videos_dir /path/to/your/video/clips
```

### Directory Structure

Your video clips directory should have the following structure:

```
/path/to/your/video/clips/
├── video_id_1.mp4
├── video_id_2.mp4
├── video_id_3.mp4
└── ...
```

Where:
- `video_id_1.mp4`, `video_id_2.mp4`, etc. are the video clip files
- The video IDs should match the folder names in your dataset's image directory
- Video files should be in MP4 format (other formats supported by OpenCV may also work)

### How It Works

1. The code extracts the video ID and frame number from the current image path in the dataset
2. It constructs a path to the corresponding video clip: `{original_videos_dir}/{video_id}.mp4`
3. It opens the video clip and seeks to the specific frame number
4. The frame is extracted and used for visualization with bounding boxes
5. Video captures are cached to avoid reopening the same video multiple times
6. If video clips aren't found, it falls back to the previous behavior

### Bounding Box Colors

- **Green**: Observed trajectory (input)
- **Red**: Ground truth future trajectory  
- **Blue**: Predicted future trajectory

### Output

The modified code will:
1. Save the bounding box coordinates as `.npz` files in `./viz_last_frame/`
2. Save visualization images with bounding boxes drawn on extracted frames as `.png` files
3. Print messages indicating which videos were opened and frames extracted
4. Automatically clean up video captures when processing is complete

## Example

```bash
# Train JAAD model with original video clips visualization
python tools/jaad/train_cvae.py \
    --data_root /path/to/jaad/dataset \
    --original_videos_dir /path/to/your/jaad/video/clips \
    --checkpoint /path/to/checkpoint.pth \
    --epochs 100
```

## Performance Features

- **Video Caching**: Videos are opened once and cached to avoid repeated file I/O operations
- **Frame Seeking**: Direct frame seeking to extract specific frames without reading through the entire video
- **Memory Management**: Video captures are automatically released when processing is complete

## Notes

- The video IDs in your clips directory must match the folder names in your dataset's image directory
- Video files should be in MP4 format (or other OpenCV-supported formats)
- Frame numbers are extracted from the dataset's image filenames (e.g., `00015.png` → frame 15)
- If video clips aren't found, the system will gracefully fall back to the previous behavior
- Make sure your video clips have the same resolution as specified in the configuration (default: 1920x1080)
- The bounding boxes are drawn using OpenCV, so the output images will be in BGR format
- The system automatically handles video cleanup to prevent memory leaks

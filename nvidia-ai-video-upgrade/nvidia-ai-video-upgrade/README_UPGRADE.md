# NVIDIA AI Video HQ upgrade

Upload the contents of this package to the root of the existing GitHub repository and choose **Commit changes**.

Required repository configuration:
- Actions secret: `NVIDIA_API_KEY`
- Actions variable: `NVIDIA_MODEL`

Then run **Actions > Generate Video with NVIDIA > Run workflow**.

Output artifact contains:
- `final_video.mp4`
- `generated_script.txt`
- `quality_report.json`

The renderer creates 1080x1920 H.264/AAC video, large Vietnamese subtitles, center-cropped images, Ken Burns motion, fast-start MP4, and automatic FFprobe quality checks.

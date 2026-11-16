#!/usr/bin/env python3
"""
Free AI Animated Video Generator
--------------------------------
Turns a plain-text script into a narrated animated video using:
- edge-tts for narration
- Pollinations.ai for scene backgrounds
- a layered character rig for blinking, mouth movement, and gestures
- FFmpeg for video encoding

Designed to run in GitHub Actions without API keys.

Usage:
    python generate_video.py \
        --script scripts/example_script.txt \
        --out output/final_video.mp4
"""

import argparse
import asyncio
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import edge_tts
import requests
from PIL import Image

DEFAULT_VOICE = "en-US-AriaNeural"
CANVAS_W, CANVAS_H = 1920, 1080
FPS = 25
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
ANIME_STYLE_SUFFIX = (
    ", anime background art, cel shaded, Studio Ghibli inspired, hand-painted anime "
    "illustration, vibrant anime color palette, no photorealism"
)
CHARACTER_DIR = Path(__file__).parent / "assets" / "character"

VISEME_MAP = {
    "A": "closed",
    "B": "closed",
    "G": "closed",
    "X": "closed",
    "C": "mid",
    "E": "mid",
    "F": "mid",
    "H": "mid",
    "D": "wide",
}


def parse_script(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    blocks = raw.split("---")

    settings = {"voice": DEFAULT_VOICE, "music": None}
    header = blocks[0].strip()

    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("VOICE:"):
            value = line.split(":", 1)[1].strip()
            if value:
                settings["voice"] = value
        elif line.upper().startswith("MUSIC:"):
            value = line.split(":", 1)[1].strip()
            settings["music"] = value or None

    scenes = []
    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue

        lines = [line for line in block.splitlines() if line.strip()]
        image_prompt = None
        narration_lines = []

        for line in lines:
            if line.upper().startswith("IMAGE:") and image_prompt is None:
                image_prompt = line.split(":", 1)[1].strip()
            else:
                narration_lines.append(line.strip())

        narration = " ".join(narration_lines).strip()
        if not narration:
            continue
        if not image_prompt:
            image_prompt = narration[:200]

        scenes.append({"image_prompt": image_prompt, "narration": narration})

    if not scenes:
        raise ValueError("No scenes found. Separate scenes with lines containing only '---'.")

    return settings, scenes


async def _synthesize(text: str, voice: str, out_mp3: Path):
    communicate = edge_tts.Communicate(text, voice)
    with out_mp3.open("wb") as output_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output_file.write(chunk["data"])


def synthesize_narration(text: str, voice: str, out_mp3: Path):
    asyncio.run(_synthesize(text, voice, out_mp3))
    if not out_mp3.exists() or out_mp3.stat().st_size < 1000:
        raise RuntimeError(f"Narration was not created correctly: {out_mp3}")


def fetch_image(prompt: str, out_path: Path, width=CANVAS_W, height=CANVAS_H, retries=4):
    seed = random.randint(1, 999_999)
    styled_prompt = f"{prompt}{ANIME_STYLE_SUFFIX}"
    encoded_prompt = quote(styled_prompt)

    urls = [
        (
            f"{POLLINATIONS_BASE}/{encoded_prompt}"
            f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
        ),
        (
            f"{POLLINATIONS_BASE}/{encoded_prompt}"
            f"?width={width}&height={height}&seed={seed}&nologo=true"
        ),
    ]

    last_error = "unknown image error"
    headers = {"User-Agent": "Mozilla/5.0 GitHub-Actions-Video-Generator/1.0"}

    for attempt in range(1, retries + 1):
        try:
            url = urls[0] if attempt < retries else urls[1]
            response = requests.get(url, timeout=180, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "image" not in content_type:
                raise RuntimeError(f"Unexpected content type: {content_type or 'missing'}")

            out_path.write_bytes(response.content)
            if out_path.stat().st_size <= 5000:
                raise RuntimeError("Image response was too small")

            with Image.open(out_path) as image:
                image.verify()
            return

        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            print(f"  [image] attempt {attempt}/{retries} failed: {last_error}")
            if attempt < retries:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to fetch image for prompt {prompt!r}: {last_error}")


def run(cmd, **kwargs):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )
    if result.returncode != 0:
        command = " ".join(str(item) for item in cmd)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n{command}\n\n"
            f"stdout:\n{result.stdout[-4000:]}\n\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    return result


def get_duration(path: Path) -> float:
    result = run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(result.stdout.strip())


def mp3_to_wav(mp3_path: Path, wav_path: Path):
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mp3_path),
        "-ar",
        "22050",
        "-ac",
        "1",
        str(wav_path),
    ])


def run_rhubarb(wav_path: Path, out_json: Path, rhubarb_bin="rhubarb"):
    try:
        run([
            rhubarb_bin,
            "-f",
            "json",
            "-o",
            str(out_json),
            "--recognizer",
            "phonetic",
            str(wav_path),
        ])
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return data.get("mouthCues", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  [lipsync] Rhubarb unavailable/failed ({exc}); using audio-volume approximation.")
        return None


def approximate_mouth_cues(wav_path: Path, duration: float):
    import audioop
    import wave

    cues = []
    try:
        with wave.open(str(wav_path), "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            chunk_ms = 90
            chunk_frames = int(frame_rate * chunk_ms / 1000)
            current_time = 0.0

            while True:
                frames = wave_file.readframes(chunk_frames)
                if not frames:
                    break

                rms = audioop.rms(frames, wave_file.getsampwidth())
                if rms < 200:
                    letter = "X"
                elif rms < 800:
                    letter = "B"
                elif rms < 2000:
                    letter = "C"
                else:
                    letter = "D"

                cues.append({
                    "start": current_time,
                    "end": current_time + chunk_ms / 1000,
                    "value": letter,
                })
                current_time += chunk_ms / 1000

    except Exception as exc:  # noqa: BLE001
        print(f"  [lipsync] approximation failed ({exc}); using a static mouth.")
        cues = [{"start": 0, "end": duration, "value": "C"}]

    return cues


def mouth_state_at(cues, current_time: float) -> str:
    for cue in cues:
        if cue["start"] <= current_time < cue["end"]:
            return VISEME_MAP.get(cue["value"], "closed")
    return "closed"


class CharacterRig:
    def __init__(self, character_dir: Path):
        required_files = [
            "rig.json",
            "base.png",
            "eye_l_open.png",
            "eye_r_open.png",
            "eye_closed.png",
            "mouth_closed.png",
            "mouth_mid.png",
            "mouth_wide.png",
            "arm_idle.png",
            "arm_raised.png",
        ]
        missing = [name for name in required_files if not (character_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Character assets are missing from {character_dir}: {', '.join(missing)}"
            )

        config = json.loads((character_dir / "rig.json").read_text(encoding="utf-8"))
        self.left_eye_pos = tuple(config["left_eye_pos"])
        self.right_eye_pos = tuple(config["right_eye_pos"])
        self.mouth_pos = tuple(config["mouth_pos"])
        self.canvas_size = tuple(config["canvas_size"])
        self.right_arm_idle_pos = tuple(config["right_arm_idle_pos"])
        self.right_arm_raised_pos = tuple(config["right_arm_raised_pos"])

        base = Image.open(character_dir / "base.png").convert("RGBA")
        eye_l_open = Image.open(character_dir / "eye_l_open.png").convert("RGBA")
        eye_r_open = Image.open(character_dir / "eye_r_open.png").convert("RGBA")
        eye_closed = Image.open(character_dir / "eye_closed.png").convert("RGBA")

        mouths = {
            "closed": Image.open(character_dir / "mouth_closed.png").convert("RGBA"),
            "mid": Image.open(character_dir / "mouth_mid.png").convert("RGBA"),
            "wide": Image.open(character_dir / "mouth_wide.png").convert("RGBA"),
        }
        arms = {
            "idle": (
                Image.open(character_dir / "arm_idle.png").convert("RGBA"),
                self.right_arm_idle_pos,
            ),
            "raised": (
                Image.open(character_dir / "arm_raised.png").convert("RGBA"),
                self.right_arm_raised_pos,
            ),
        }

        self.frames = {}
        for eyes_state in ("open", "closed"):
            for mouth_state, mouth_image in mouths.items():
                for arm_state, (arm_image, arm_position) in arms.items():
                    frame = base.copy()
                    frame.paste(arm_image, arm_position, arm_image)

                    if eyes_state == "open":
                        frame.paste(eye_l_open, self.left_eye_pos, eye_l_open)
                        frame.paste(eye_r_open, self.right_eye_pos, eye_r_open)
                    else:
                        frame.paste(eye_closed, self.left_eye_pos, eye_closed)
                        frame.paste(eye_closed, self.right_eye_pos, eye_closed)

                    frame.paste(mouth_image, self.mouth_pos, mouth_image)
                    self.frames[(eyes_state, mouth_state, arm_state)] = frame

    def get(self, eyes_state: str, mouth_state: str, arm_state: str) -> Image.Image:
        return self.frames[(eyes_state, mouth_state, arm_state)]


def make_blink_schedule(duration: float, min_gap=2.5, max_gap=5.5, blink_len=0.14):
    schedule = []
    current_time = random.uniform(1.0, min_gap)
    while current_time < duration:
        schedule.append((current_time, current_time + blink_len))
        current_time += random.uniform(min_gap, max_gap)
    return schedule


def is_blinking(schedule, current_time: float) -> bool:
    return any(start <= current_time < end for start, end in schedule)


def make_gesture_schedule(duration: float, min_gap=3.5, max_gap=7.0, gesture_len=1.3):
    schedule = []
    current_time = random.uniform(1.5, min_gap)
    while current_time < duration - gesture_len * 0.5:
        schedule.append((current_time, current_time + gesture_len))
        current_time += random.uniform(min_gap, max_gap)
    return schedule


def arm_state_at(schedule, current_time: float) -> str:
    for start, end in schedule:
        if start <= current_time < end:
            return "raised"
    return "idle"


def prepare_background(image_path: Path, width, height) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    source_ratio = image.width / image.height
    target_ratio = width / height

    if source_ratio > target_ratio:
        new_height = height
        new_width = int(height * source_ratio)
    else:
        new_width = width
        new_height = int(width / source_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def available_video_encoder():
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    encoders = result.stdout or ""
    if "libx264" in encoders:
        return "libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    if " h264 " in encoders:
        return "h264", ["-c:v", "h264"]
    if "mpeg4" in encoders:
        return "mpeg4", ["-c:v", "mpeg4", "-q:v", "4"]
    raise RuntimeError("FFmpeg has no supported H.264 or MPEG-4 video encoder.")


def render_scene(
    background: Image.Image,
    rig: CharacterRig,
    mouth_cues,
    blink_schedule,
    gesture_schedule,
    duration: float,
    audio_path: Path,
    out_path: Path,
    char_scale=0.85,
    fps=FPS,
):
    if duration <= 0:
        raise ValueError(f"Invalid scene duration: {duration}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    char_h = int(CANVAS_H * char_scale)
    char_w = int(char_h * rig.canvas_size[0] / rig.canvas_size[1])
    char_h = max(2, char_h - char_h % 2)
    char_w = max(2, char_w - char_w % 2)

    anchor_x = (CANVAS_W - char_w) // 2
    anchor_y = CANVAS_H - char_h + 20

    resized_cache = {
        key: image.resize((char_w, char_h), Image.LANCZOS)
        for key, image in rig.frames.items()
    }

    total_frames = max(int(math.ceil(duration * fps)), fps)
    encoder_name, encoder_args = available_video_encoder()
    print(f"     FFmpeg video encoder: {encoder_name}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{CANVAS_W}x{CANVAS_H}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    background_rgba = background.convert("RGBA")
    frames_written = 0
    pipe_failed = False

    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

        try:
            if process.stdin is None:
                raise RuntimeError("Could not open FFmpeg stdin.")

            for frame_index in range(total_frames):
                if process.poll() is not None:
                    pipe_failed = True
                    break

                current_time = frame_index / fps
                mouth_state = mouth_state_at(mouth_cues, current_time) if mouth_cues else "closed"
                eyes_state = "closed" if is_blinking(blink_schedule, current_time) else "open"
                arm_state = arm_state_at(gesture_schedule, current_time)
                character_image = resized_cache[(eyes_state, mouth_state, arm_state)]

                bob = int(5 * math.sin(2 * math.pi * current_time / 2.6))
                frame = background_rgba.copy()
                frame.alpha_composite(character_image, (anchor_x, anchor_y + bob))

                try:
                    process.stdin.write(frame.convert("RGB").tobytes())
                    frames_written += 1
                except BrokenPipeError:
                    pipe_failed = True
                    break

        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            try:
                return_code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()

            stderr_file.seek(0)
            ffmpeg_error = stderr_file.read().decode("utf-8", errors="replace").strip()

    if return_code != 0:
        command_text = " ".join(str(item) for item in ffmpeg_cmd)
        raise RuntimeError(
            "FFmpeg scene rendering failed.\n"
            f"Return code: {return_code}\n"
            f"Frames written: {frames_written}/{total_frames}\n"
            f"Broken pipe detected: {pipe_failed}\n"
            f"Encoder: {encoder_name}\n"
            f"Output: {out_path}\n"
            f"Command: {command_text}\n\n"
            f"FFmpeg error:\n{ffmpeg_error[-8000:] or 'No FFmpeg error text was produced.'}"
        )

    if not out_path.exists():
        raise RuntimeError(f"FFmpeg exited successfully but did not create {out_path}")
    if out_path.stat().st_size < 10_000:
        raise RuntimeError(
            f"FFmpeg created an unexpectedly small video: {out_path} "
            f"({out_path.stat().st_size} bytes)"
        )

    print(
        f"     Scene rendered successfully: {frames_written} frames, "
        f"{out_path.stat().st_size / 1024 / 1024:.1f} MB"
    )


def concat_clips(clip_paths, out_path: Path, workdir: Path):
    if not clip_paths:
        raise ValueError("No scene clips were created.")

    list_file = workdir / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{path.resolve()}'" for path in clip_paths),
        encoding="utf-8",
    )

    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ])


def add_background_music(video_path: Path, music_path: Path, out_path: Path, volume=0.12):
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-filter_complex",
        (
            f"[1:a]volume={volume}[music];"
            "[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        ),
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Generate an animated talking-character video from a script."
    )
    parser.add_argument("--script", required=True)
    parser.add_argument("--out", default="output/final_video.mp4")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--music", default=None)
    parser.add_argument("--character-scale", type=float, default=0.85)
    parser.add_argument("--rhubarb-bin", default="rhubarb")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH.")

    settings, scenes = parse_script(args.script)
    voice = args.voice or settings["voice"]
    music = args.music or settings["music"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = out_path.parent / "_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    print(f"Loaded {len(scenes)} scene(s). Voice: {voice}. Music: {music or 'none'}")
    rig = CharacterRig(CHARACTER_DIR)

    clip_paths = []
    for scene_index, scene in enumerate(scenes):
        print(f"\n[{scene_index + 1}/{len(scenes)}] {scene['narration'][:70]}...")

        audio_mp3 = workdir / f"scene_{scene_index:03}.mp3"
        audio_wav = workdir / f"scene_{scene_index:03}.wav"
        image_path = workdir / f"scene_{scene_index:03}.jpg"
        rhubarb_json = workdir / f"scene_{scene_index:03}_mouth.json"
        clip_path = workdir / f"scene_{scene_index:03}.mp4"

        print("  -> generating narration (edge-tts)")
        synthesize_narration(scene["narration"], voice, audio_mp3)
        duration = get_duration(audio_mp3) + 0.3
        mp3_to_wav(audio_mp3, audio_wav)

        print("  -> computing lip-sync (Rhubarb)")
        mouth_cues = run_rhubarb(audio_wav, rhubarb_json, args.rhubarb_bin)
        if mouth_cues is None:
            mouth_cues = approximate_mouth_cues(audio_wav, duration)

        blink_schedule = make_blink_schedule(duration)
        gesture_schedule = make_gesture_schedule(duration)

        print(f"  -> generating background: {scene['image_prompt'][:70]}...")
        fetch_image(scene["image_prompt"], image_path, CANVAS_W, CANVAS_H)
        background = prepare_background(image_path, CANVAS_W, CANVAS_H)

        print(f"  -> rendering animated scene ({duration:.1f}s, {math.ceil(duration * FPS)} frames)")
        render_scene(
            background,
            rig,
            mouth_cues,
            blink_schedule,
            gesture_schedule,
            duration,
            audio_mp3,
            clip_path,
            char_scale=args.character_scale,
        )
        clip_paths.append(clip_path)

    print("\nConcatenating all scenes...")
    no_music_path = workdir / "combined_no_music.mp4"
    concat_clips(clip_paths, no_music_path, workdir)

    if music and Path(music).exists():
        print("Mixing background music...")
        add_background_music(no_music_path, Path(music), out_path)
    else:
        if music:
            print(f"  Music file '{music}' was not found; continuing without music.")
        shutil.copy2(no_music_path, out_path)

    total_duration = get_duration(out_path)
    print(f"\nDone! Video saved to: {out_path} ({total_duration:.1f}s)")

    if not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

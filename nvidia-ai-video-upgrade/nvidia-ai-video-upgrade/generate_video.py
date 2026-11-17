import argparse
import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import edge_tts
import requests

WIDTH = 1080
HEIGHT = 1920
FPS = 30


def run(cmd):
    print('+', ' '.join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def ffprobe_duration(path):
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
    ], text=True).strip()
    return max(0.25, float(out))


def parse_script(path):
    raw = Path(path).read_text(encoding='utf-8').replace('\r\n', '\n')
    voice_match = re.search(r'(?mi)^\s*VOICE\s*:\s*(.+?)\s*$', raw)
    music_match = re.search(r'(?mi)^\s*MUSIC\s*:\s*(.*?)\s*$', raw)
    voice = voice_match.group(1).strip() if voice_match else 'vi-VN-HoaiMyNeural'
    music = music_match.group(1).strip() if music_match else ''
    raw = re.sub(r'(?mi)^\s*(VOICE|MUSIC)\s*:.*$', '', raw)
    chunks = [x.strip() for x in re.split(r'(?m)^\s*---\s*$', raw) if x.strip()]
    scenes = []
    for chunk in chunks:
        im = re.search(r'(?mi)^\s*IMAGE\s*:\s*(.+?)\s*$', chunk)
        prompt = im.group(1).strip() if im else ''
        narration = re.sub(r'(?mi)^\s*IMAGE\s*:.*$', '', chunk).strip()
        if narration:
            scenes.append({'image_prompt': prompt or narration, 'narration': narration})
    if not scenes:
        raise ValueError('No valid scenes found in script')
    return voice, music, scenes


async def synthesize(text, voice, audio_path, timing_path, rate):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    words = []
    with open(audio_path, 'wb') as audio:
        async for item in communicate.stream():
            if item['type'] == 'audio':
                audio.write(item['data'])
            elif item['type'] == 'WordBoundary':
                words.append({
                    'text': item['text'],
                    'start': item['offset'] / 10_000_000,
                    'end': (item['offset'] + item['duration']) / 10_000_000,
                })
    Path(timing_path).write_text(json.dumps(words, ensure_ascii=False), encoding='utf-8')
    return words


def download_image(prompt, output, seed):
    enhanced = (
        f'{prompt}, vertical 9:16 composition, cinematic lighting, photorealistic, '
        'high detail, subject centered, no text, no logo, no watermark'
    )
    url = f'https://image.pollinations.ai/prompt/{quote(enhanced)}?width=1080&height=1920&seed={seed}&nologo=true'
    headers = {'User-Agent': 'Mozilla/5.0 AI-video-generator'}
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=180)
            response.raise_for_status()
            if len(response.content) < 10_000:
                raise RuntimeError('Image response is unexpectedly small')
            Path(output).write_bytes(response.content)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f'Image generation failed after retries: {last_error}')


def ass_escape(text):
    return text.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}').replace('\n', r'\N')


def ass_time(seconds):
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h}:{m:02d}:{s:05.2f}'


def group_words(words, max_words=5, max_seconds=1.9):
    groups, current = [], []
    for w in words:
        if current and (len(current) >= max_words or w['end'] - current[0]['start'] > max_seconds):
            groups.append(current)
            current = []
        current.append(w)
        if re.search(r'[.!?,;:]$', w['text']) and len(current) >= 2:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def write_ass(words, duration, output):
    header = f'''[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Shorts,Arial,74,&H00FFFFFF,&H0000D7FF,&H00101010,&H70000000,-1,0,0,0,100,100,0,0,1,6,2,2,70,70,410,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines = []
    if words:
        for group in group_words(words):
            start = group[0]['start']
            end = min(duration, max(group[-1]['end'] + 0.12, start + 0.55))
            text = ass_escape(' '.join(x['text'] for x in group))
            lines.append(f'Dialogue: 0,{ass_time(start)},{ass_time(end)},Shorts,,0,0,0,,{text}')
    Path(output).write_text(header + '\n'.join(lines) + '\n', encoding='utf-8')


def render_scene(image, audio, subtitles, output, index):
    duration = ffprobe_duration(audio)
    frames = max(1, math.ceil(duration * FPS))
    zoom_expr = "min(zoom+0.0007,1.10)" if index % 2 == 0 else "min(zoom+0.0005,1.08)"
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"subtitles={subtitles}:fontsdir=/usr/share/fonts/truetype/dejavu,"
        "format=yuv420p"
    )
    run([
        'ffmpeg', '-y', '-loop', '1', '-i', image, '-i', audio,
        '-vf', vf, '-t', f'{duration:.3f}', '-r', str(FPS),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
        '-movflags', '+faststart', '-shortest', output
    ])


def concat_scenes(scene_files, output, work):
    concat_file = Path(work) / 'concat.txt'
    concat_file.write_text(''.join(f"file '{Path(p).resolve()}'\n" for p in scene_files), encoding='utf-8')
    run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
        '-movflags', '+faststart', '-pix_fmt', 'yuv420p', output
    ])


def mix_music(video, music, output):
    run([
        'ffmpeg', '-y', '-i', video, '-stream_loop', '-1', '-i', music,
        '-filter_complex',
        '[0:a]loudnorm=I=-14:TP=-1:LRA=11[voice];'
        '[1:a]volume=0.10,afade=t=in:st=0:d=0.8[music];'
        '[voice][music]amix=inputs=2:duration=first:dropout_transition=2[a]',
        '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac',
        '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', '-shortest', output
    ])


def verify_output(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)
    ], text=True))
    videos = [s for s in data['streams'] if s.get('codec_type') == 'video']
    audios = [s for s in data['streams'] if s.get('codec_type') == 'audio']
    if not videos or not audios:
        raise RuntimeError('QC failed: missing video or audio stream')
    v = videos[0]
    if int(v.get('width', 0)) != WIDTH or int(v.get('height', 0)) != HEIGHT:
        raise RuntimeError(f"QC failed: output is {v.get('width')}x{v.get('height')}")
    if Path(path).stat().st_size < 500_000:
        raise RuntimeError('QC failed: output file is too small')
    return {
        'width': int(v['width']), 'height': int(v['height']),
        'video_codec': v.get('codec_name'), 'audio_codec': audios[0].get('codec_name'),
        'duration_seconds': round(float(data['format']['duration']), 2),
        'size_bytes': Path(path).stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', required=True)
    parser.add_argument('--out', default='output/final_video.mp4')
    parser.add_argument('--voice')
    parser.add_argument('--music')
    parser.add_argument('--character-scale', type=float, default=1.0)
    parser.add_argument('--rhubarb-bin')
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--rate', default='+8%')
    args = parser.parse_args()

    file_voice, script_music, scenes = parse_script(args.script)
    voice = args.voice or file_voice
    music = args.music or script_music
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='nvidia_video_'))
    print(f'Working directory: {work}')
    scene_files = []

    try:
        for i, scene in enumerate(scenes, 1):
            prefix = work / f'scene_{i:02d}'
            audio = str(prefix) + '.mp3'
            timing = str(prefix) + '.json'
            image = str(prefix) + '.jpg'
            ass = str(prefix) + '.ass'
            clip = str(prefix) + '.mp4'
            print(f'Building scene {i}/{len(scenes)}')
            words = asyncio.run(synthesize(scene['narration'], voice, audio, timing, args.rate))
            duration = ffprobe_duration(audio)
            write_ass(words, duration, ass)
            download_image(scene['image_prompt'], image, random.randint(1, 2_000_000_000))
            render_scene(image, audio, ass, clip, i)
            scene_files.append(clip)

        joined = work / 'joined.mp4'
        concat_scenes(scene_files, joined, work)
        if music and Path(music).is_file():
            mix_music(joined, music, out)
        else:
            shutil.copy2(joined, out)
        report = verify_output(out)
        report_path = out.with_name('quality_report.json')
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f'Finished video: {out}')
    finally:
        if args.keep_temp:
            print(f'Temporary files retained: {work}')
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()

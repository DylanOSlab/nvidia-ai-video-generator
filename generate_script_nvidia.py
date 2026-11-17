import argparse
import os
import re
import sys
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def build_prompt(
    topic: str,
    language: str,
    voice: str,
    scene_count: int,
    duration: int,
    style: str,
) -> str:
    return f"""
Create a complete script for a vertical short-form video.

TOPIC:
{topic}

SETTINGS:
- Narration language: {language}
- Number of scenes: exactly {scene_count}
- Approximate total duration: {duration} seconds
- Visual style: {style}
- Audience: general social-media audience
- Format: vertical 9:16 video

OUTPUT FORMAT:
Return plain text only, using exactly this structure:

VOICE: {voice}
MUSIC:

---

IMAGE: detailed English image prompt, vertical 9:16 composition, no text, no logo
Narration in {language}.

---

IMAGE: detailed English image prompt, vertical 9:16 composition, no text, no logo
Narration in {language}.

RULES:
1. Return exactly {scene_count} scenes.
2. Begin scene 1 with a strong hook.
3. Write all IMAGE prompts in English.
4. Write all narration in {language}.
5. Keep each scene concise and spoken naturally.
6. Make scenes visually different but narratively coherent.
7. Never put written words, captions, watermarks, or logos in image prompts.
8. Do not return Markdown fences.
9. Do not add explanations before or after the script.
10. Do not invent exact statistics, quotations, or current events.
""".strip()


def validate_script(script: str, expected_scenes: int) -> None:
    if not script.strip():
        raise ValueError("NVIDIA returned an empty response")

    image_count = len(
        re.findall(r"(?mi)^\s*IMAGE\s*:", script)
    )

    if image_count != expected_scenes:
        raise ValueError(
            f"Invalid number of scenes: expected {expected_scenes}, "
            f"received {image_count}"
        )

    if not re.search(r"(?mi)^\s*VOICE\s*:", script):
        raise ValueError("The generated script does not contain VOICE:")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--language", default="Vietnamese")
    parser.add_argument("--voice", default="vi-VN-HoaiMyNeural")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument(
        "--style",
        default="cinematic, realistic, modern, high detail",
    )
    parser.add_argument(
        "--output",
        default="scripts/generated_script.txt",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    model = os.environ.get("NVIDIA_MODEL")

    if not api_key:
        raise RuntimeError(
            "Missing NVIDIA_API_KEY in GitHub Actions Secrets"
        )

    if not model:
        raise RuntimeError(
            "Missing NVIDIA_MODEL in GitHub Actions Variables"
        )

    scene_count = max(3, min(args.scenes, 10))
    duration = max(20, min(args.duration, 120))

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )

    prompt = build_prompt(
        topic=args.topic,
        language=args.language,
        voice=args.voice,
        scene_count=scene_count,
        duration=duration,
        style=args.style,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional short-video scriptwriter. "
                        "Follow the requested output format exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=2200,
            stream=False,
        )
    except APIStatusError as exc:
        print(
            f"NVIDIA API error: HTTP {exc.status_code}",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        raise
    except APIConnectionError as exc:
        print(
            "Could not connect to the NVIDIA API",
            file=sys.stderr,
        )
        raise exc

    script = response.choices[0].message.content or ""
    script = script.strip()

    validate_script(script, scene_count)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script + "\n", encoding="utf-8")

    print(f"NVIDIA model: {model}")
    print(f"Generated scenes: {scene_count}")
    print(f"Script saved to: {output}")


if __name__ == "__main__":
    main()

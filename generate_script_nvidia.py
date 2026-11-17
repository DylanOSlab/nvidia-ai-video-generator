import argparse
import os
import re
import sys
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MIN_VALID_SCENES = 2


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
2. Every scene must contain exactly one IMAGE: line.
3. Separate every scene with a line containing only ---.
4. Begin scene 1 with a strong hook.
5. Write all IMAGE prompts in English.
6. Write all narration in {language}.
7. Keep each scene concise and natural when spoken.
8. Make scenes visually different but narratively coherent.
9. Never request written words, captions, watermarks, or logos in images.
10. Do not return Markdown fences.
11. Do not add explanations before or after the script.
12. Do not invent exact statistics, quotations, or current events.
""".strip()


def clean_model_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def count_scenes(script: str) -> int:
    return len(re.findall(r"(?mi)^\s*IMAGE\s*:", script))


def validate_script(script: str, expected_scenes: int) -> int:
    if not script.strip():
        raise ValueError("NVIDIA returned an empty response")

    image_count = count_scenes(script)

    if image_count < MIN_VALID_SCENES:
        raise ValueError(
            f"Not enough valid scenes: received {image_count}; "
            f"minimum is {MIN_VALID_SCENES}"
        )

    if not re.search(r"(?mi)^\s*VOICE\s*:", script):
        raise ValueError("The generated script does not contain VOICE:")

    if image_count != expected_scenes:
        print(
            f"Warning: requested {expected_scenes} scenes, "
            f"but NVIDIA generated {image_count}. "
            "Continuing with the available valid scenes.",
            file=sys.stderr,
        )

    return image_count


def request_completion(
    client: OpenAI,
    model: str,
    system_message: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        stream=False,
    )

    if not response.choices:
        raise ValueError("NVIDIA returned no completion choices")

    return clean_model_output(response.choices[0].message.content or "")


def repair_script(
    client: OpenAI,
    model: str,
    script: str,
    expected_scenes: int,
    language: str,
    voice: str,
) -> str:
    actual_scenes = count_scenes(script)
    repair_prompt = f"""
The video script below contains {actual_scenes} IMAGE scenes, but it must contain exactly {expected_scenes} scenes.

Rewrite the complete script so that it contains exactly {expected_scenes} scenes.

Required format:
VOICE: {voice}
MUSIC:

---

IMAGE: detailed English image prompt, vertical 9:16 composition, no text, no logo
Narration in {language}.

Rules:
- Return only the complete corrected script.
- Include exactly {expected_scenes} IMAGE: lines.
- Separate every scene with a line containing only ---.
- IMAGE prompts must be in English.
- Narration must be in {language}.
- Preserve the original topic and core message.
- Do not use Markdown fences.
- Do not add an explanation.

SCRIPT TO CORRECT:

{script}
""".strip()

    return request_completion(
        client=client,
        model=model,
        system_message=(
            "You repair structured video scripts. Follow the requested "
            "plain-text format and exact scene count."
        ),
        user_message=repair_prompt,
        temperature=0.2,
        max_tokens=3000,
    )


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
        raise RuntimeError("Missing NVIDIA_API_KEY in GitHub Actions Secrets")

    if not model:
        raise RuntimeError("Missing NVIDIA_MODEL in GitHub Actions Variables")

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
        script = request_completion(
            client=client,
            model=model,
            system_message=(
                "You are a professional short-video scriptwriter. "
                "Follow the requested plain-text format exactly."
            ),
            user_message=prompt,
            temperature=0.6,
            max_tokens=2600,
        )

        first_count = count_scenes(script)
        print(f"Initial NVIDIA response contains {first_count} IMAGE scenes")

        if first_count != scene_count:
            print(
                f"Requested {scene_count} scenes but received {first_count}. "
                "Attempting one automatic repair."
            )
            repaired = repair_script(
                client=client,
                model=model,
                script=script,
                expected_scenes=scene_count,
                language=args.language,
                voice=args.voice,
            )
            repaired_count = count_scenes(repaired)
            print(f"Repaired response contains {repaired_count} IMAGE scenes")

            if repaired_count >= MIN_VALID_SCENES:
                script = repaired

        actual_scene_count = validate_script(script, scene_count)

    except APIStatusError as exc:
        print(
            f"NVIDIA API error: HTTP {exc.status_code}",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        raise
    except APIConnectionError as exc:
        print("Could not connect to the NVIDIA API", file=sys.stderr)
        raise exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script.rstrip() + "\n", encoding="utf-8")

    print(f"NVIDIA model: {model}")
    print(f"Requested scenes: {scene_count}")
    print(f"Final valid scenes: {actual_scene_count}")
    print(f"Script saved to: {output}")


if __name__ == "__main__":
    main()

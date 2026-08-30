"""Download real game-capture frames to use as the clean source set.

Pulls from ``Bingsu/Gameplay_Images`` on the Hugging Face Hub (CC-BY-4.0): frames
sampled from gameplay footage of ten titles — Among Us, Apex Legends, Fortnite,
Forza Horizon, Free Fire, Genshin Impact, God of War, Minecraft, Roblox and
Terraria.  Real render output, so the detector learns fault signatures against
the lighting, HUDs, aliasing and post-processing it would actually meet.

Frames are requested at spread offsets rather than sequentially, because the
source is ordered by title — striding keeps all ten games represented.

Usage::

    python scripts/fetch_game_frames.py                  # 1500 frames
    python scripts/fetch_game_frames.py --count 400 --output-dir data/small
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

DATASET_ID: str = "Bingsu/Gameplay_Images"
ROWS_ENDPOINT: str = "https://datasets-server.huggingface.co/rows"
DATASET_ROWS: int = 10_000
ROWS_PER_REQUEST: int = 50          # the endpoint's practical page size
DEFAULT_COUNT: int = 1500
DEFAULT_OUTPUT: Path = config.DATA_ROOT / "game_frames"
DOWNLOAD_WORKERS: int = 8
REQUEST_TIMEOUT: int = 90
LIST_ATTEMPTS: int = 5
LIST_BACKOFF_SECONDS: float = 4.0
ATTRIBUTION_FILENAME: str = "SOURCE.txt"


def fetch_json(url: str, attempts: int = LIST_ATTEMPTS) -> dict:
    """GET a URL and parse the response as JSON, backing off on rate limits.

    The rows endpoint throttles aggressively once a listing run gets going, and
    a throttled page is a page of frames lost — so retry rather than skip.

    Raises:
        urllib.error.URLError: If every attempt fails.
    """
    delay = LIST_BACKOFF_SECONDS
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2
    raise urllib.error.URLError("exhausted retries")  # pragma: no cover


def list_rows(count: int) -> list[tuple[str, str]]:
    """Return ``(image_url, label)`` pairs spread across the whole dataset.

    Raises:
        RuntimeError: If the rows endpoint returns nothing usable.
    """
    pages = max(1, -(-count // ROWS_PER_REQUEST))
    stride = max(1, DATASET_ROWS // pages)
    collected: list[tuple[str, str]] = []
    label_names: list[str] = []

    for page in range(pages):
        offset = min(page * stride, DATASET_ROWS - ROWS_PER_REQUEST)
        query = urllib.parse.urlencode({
            "dataset": DATASET_ID, "config": "default", "split": "train",
            "offset": offset, "length": ROWS_PER_REQUEST,
        })
        try:
            payload = fetch_json(f"{ROWS_ENDPOINT}?{query}")
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  [warn] page at offset {offset} gave up: {error}")
            continue

        if not label_names:
            for feature in payload.get("features", []):
                if feature["name"] == "label":
                    label_names = feature["type"].get("names", [])

        for entry in payload.get("rows", []):
            row = entry["row"]
            source = row.get("image", {}).get("src")
            if not source:
                continue
            index = row.get("label")
            name = (
                label_names[index] if isinstance(index, int) and index < len(label_names)
                else "unknown"
            )
            collected.append((source, name.replace(" ", "_").lower()))

        print(f"  listed {len(collected):>5} / {count} frames", end="\r")

    print()
    if not collected:
        raise RuntimeError(
            "the Hugging Face rows endpoint returned no images; "
            "check network access and that the dataset is still public"
        )
    return collected[:count]


def download_one(job: tuple[int, str, str], output_dir: Path) -> str | None:
    """Fetch one frame and save it as RGB PNG. Returns the label, or None."""
    index, url, label = job
    destination = output_dir / f"{label}_{index:05d}.png"
    if destination.exists():
        return label
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
        with Image.open(io.BytesIO(raw)) as image:
            image.convert("RGB").save(destination, format="PNG")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        print(f"  [warn] frame {index} failed: {type(error).__name__}: {error}")
        return None
    return label


def write_attribution(output_dir: Path, saved: int) -> None:
    """Record where the frames came from, next to the frames themselves."""
    (output_dir / ATTRIBUTION_FILENAME).write_text(
        f"{saved} frames from the Hugging Face dataset '{DATASET_ID}'\n"
        f"https://huggingface.co/datasets/{DATASET_ID}\n"
        "Licence: CC-BY-4.0. Frames sampled from gameplay footage of ten titles.\n"
        "Downloaded by scripts/fetch_game_frames.py.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the fetch CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="how many frames to download")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download the frames; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.count < 1:
        print(f"error: --count must be >= 1, got {args.count}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"listing frames from {DATASET_ID}")

    try:
        rows = list_rows(args.count)
    except (RuntimeError, urllib.error.URLError) as error:
        print(f"error: {error}")
        return 2

    print(f"downloading {len(rows)} frames into {output_dir}")
    jobs = [(index, url, label) for index, (url, label) in enumerate(rows)]
    counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, label in enumerate(
            pool.map(lambda job: download_one(job, output_dir), jobs), start=1
        ):
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", end="\r")

    saved = sum(counts.values())
    print(f"\nsaved {saved} frames")
    for label, total in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {label:<18}{total:>5}")

    if saved == 0:
        print("error: nothing downloaded")
        return 2

    write_attribution(output_dir, saved)
    print(f"\nnow train with:\n  py -m src.train --clean-dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inline the figures in ``visuals/`` into a single self-contained HTML report.

The template refers to images as ``{{IMG:name}}``; each placeholder is replaced
with a ``data:`` URI holding that file's bytes, so the finished page carries its
own images and can be published or emailed as one file.

Usage::

    python scripts/build_report.py
    python scripts/build_report.py --output docs/index.html --standalone
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

TEMPLATE_PATH: Path = Path(__file__).resolve().parent / "report_v2_template.html"
VISUAL_DIR: Path = config.PROJECT_ROOT / "visuals"
DEFAULT_OUTPUT: Path = VISUAL_DIR / "report_v2.html"

PLACEHOLDER = re.compile(r"\{\{IMG:([A-Za-z0-9_\-]+)\}\}")
STYLE_PLACEHOLDER: str = "{{STYLE}}"
STYLE_PATH: Path = Path(__file__).resolve().parent / "report_style.css"

#: Document skeleton for hosting the report anywhere other than as an Artifact.
#: The Artifact host wraps published HTML in exactly this much of a shell — a
#: charset, a viewport, and a minimal reset — so a standalone copy has to supply
#: it or the page renders with the browser's default body margin instead.
STANDALONE_HEAD: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light; }
  body { margin: 0; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
"""
STANDALONE_TAIL: str = "\n</body>\n</html>\n"
MIME_TYPES: dict[str, str] = {".png": "image/png", ".gif": "image/gif",
                              ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
#: Published artifacts are capped at 16MB including inlined data.
SIZE_LIMIT_BYTES: int = 16 * 1024 * 1024


def find_figure(name: str) -> Path:
    """Locate ``name`` in ``visuals/`` regardless of image extension.

    Raises:
        FileNotFoundError: If no supported image file matches.
    """
    for suffix in MIME_TYPES:
        candidate = VISUAL_DIR / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no figure named {name!r} in {VISUAL_DIR}. "
        "Run scripts/make_visuals.py and scripts/live_view.py --gif first."
    )


def data_uri(path: Path) -> str:
    """Encode a file as a ``data:`` URI."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{MIME_TYPES[path.suffix.lower()]};base64,{encoded}"


#: Elements a template writes at the top of its body that belong in <head>.
HEAD_ELEMENTS = re.compile(
    r"<title>.*?</title>|<style>.*?</style>|<link\b[^>]*>", re.S | re.I
)


def wrap_standalone(body: str) -> str:
    """Wrap rendered report HTML in a complete, valid HTML document.

    Templates are authored for the Artifact runtime, which supplies the document
    shell, so they open with a bare ``<title>``/``<style>``/``<link>`` run. Those
    are hoisted into a real ``<head>`` here rather than left in the body for the
    browser's error recovery to sort out.
    """
    head_parts: list[str] = []

    def lift(match: re.Match[str]) -> str:
        head_parts.append(match.group(0))
        return ""

    stripped = HEAD_ELEMENTS.sub(lift, body, count=0).strip()
    head = "\n".join(head_parts)
    return f"{STANDALONE_HEAD}{head}\n</head>\n<body>\n{stripped}{STANDALONE_TAIL}"


def build(template_path: Path, output_path: Path, standalone: bool = False) -> int:
    """Render the template with every image inlined; returns an exit code.

    Args:
        template_path: Report template containing ``{{STYLE}}`` and ``{{IMG:*}}``.
        output_path: File to write.
        standalone: Wrap the result in a full HTML document, for hosting outside
            the Artifact runtime (GitHub Pages, a plain web server, email).
    """
    if not template_path.is_file():
        print(f"error: template not found: {template_path}")
        return 2

    template = template_path.read_text(encoding="utf-8")
    # One stylesheet shared by every report template, inlined at build time.
    if STYLE_PLACEHOLDER in template:
        if not STYLE_PATH.is_file():
            print(f"error: stylesheet not found: {STYLE_PATH}")
            return 2
        stylesheet = STYLE_PATH.read_text(encoding="utf-8")
        template = template.replace(
            STYLE_PLACEHOLDER, f"<style>\n{stylesheet}\n</style>"
        )
    used: list[tuple[str, int]] = []

    def replace(match: re.Match[str]) -> str:
        path = find_figure(match.group(1))
        used.append((path.name, path.stat().st_size))
        return data_uri(path)

    try:
        rendered = PLACEHOLDER.sub(replace, template)
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 2

    if not used:
        print(f"error: template has no {{{{IMG:...}}}} placeholders")
        return 2

    if standalone:
        rendered = wrap_standalone(rendered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    size = output_path.stat().st_size
    for name, raw in sorted(used, key=lambda item: -item[1]):
        print(f"  inlined {name:<28} {raw / 1024:8.0f} KB")
    print(f"wrote {output_path} ({size / 1024 / 1024:.2f} MB)")

    if size > SIZE_LIMIT_BYTES:
        print(f"warning: exceeds the {SIZE_LIMIT_BYTES / 1024 / 1024:.0f}MB publish limit")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the report CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--standalone", action="store_true",
                        help="wrap in a full HTML document for hosting outside "
                             "the Artifact runtime, e.g. GitHub Pages")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    return build(Path(args.template), Path(args.output), args.standalone)


if __name__ == "__main__":
    raise SystemExit(main())

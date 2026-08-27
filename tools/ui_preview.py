"""Show the overlay with no audio hardware and no providers.

    python tools/ui_preview.py            interactive
    python tools/ui_preview.py --shot out/  render PNGs and exit

Needs only PySide6:

    pip install PySide6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def shoot(outdir: Path) -> int:
    """Render each widget and each status colour to PNG, then exit.

    Runs headless, so it works over a remote session and in CI - which makes
    the overlay's appearance reviewable without a desktop.
    """
    from PySide6.QtWidgets import QApplication

    from app.models.state import Colour, Mode
    from app.views.demo import STATES, DemoController

    outdir.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    demo = DemoController(auto=False)
    demo.bubble.show()
    demo.panel.show()
    demo.transcript.show()
    demo.transcript.resize(520, 380)
    for _ in range(6):
        demo.next_line()
    app.processEvents()

    for i, (mode, colour, _) in enumerate(STATES):
        demo._set(i)
        app.processEvents()
        demo.bubble.grab().save(str(outdir / f"bubble-{colour.value}.png"))
        demo.panel.grab().save(str(outdir / f"panel-{mode.value}.png"))
    demo.transcript.grab().save(str(outdir / "transcript.png"))

    print(f"  wrote {len(list(outdir.glob('*.png')))} images to {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shot", metavar="DIR", help="render PNGs headlessly and exit")
    args = ap.parse_args()

    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("\n  PySide6 is not installed.  pip install PySide6\n")
        return 2

    if args.shot:
        return shoot(Path(args.shot))

    from app.views.demo import main as demo_main

    print(
        "\n  Overlay preview — no audio, no providers, scripted data.\n"
        "  Click the bubble to open the panel. States cycle every 7s.\n"
        "  Close the transcript window to quit.\n"
    )
    return demo_main()


if __name__ == "__main__":
    raise SystemExit(main())

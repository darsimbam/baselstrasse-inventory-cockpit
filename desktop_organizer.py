"""
Desktop Organizer - Groups files and folders by modification date (YYYY-MM).
Run with --dry-run to preview changes before applying them.
"""

import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path


def get_desktop_path() -> Path:
    """Resolve the desktop path, accounting for OneDrive redirection."""
    candidates = [
        Path.home() / "OneDrive" / "Desktop",
        Path.home() / "Desktop",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate Desktop folder.")


def get_date_label(path: Path) -> str:
    """Return 'YYYY-MM' based on the item's modification time."""
    mtime = path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m")


def organize(dry_run: bool = False) -> None:
    desktop = get_desktop_path()
    print(f"Desktop: {desktop}\n")

    items = [p for p in desktop.iterdir() if p.name != "desktop.ini"]

    # Collect moves first so we don't move date-folders we're about to create
    moves: list[tuple[Path, Path]] = []
    date_folders: set[str] = set()

    for item in items:
        label = get_date_label(item)
        date_folders.add(label)
        dest_dir = desktop / label
        dest = dest_dir / item.name
        moves.append((item, dest))

    # Filter out items that ARE the target date folders themselves
    # (e.g. if a "2025-03" folder already exists on the desktop)
    moves = [(src, dst) for src, dst in moves if src.name not in date_folders]

    if not moves:
        print("Nothing to organize.")
        return

    print(f"{'DRY RUN — ' if dry_run else ''}{'Items to move':30s}  ->  Destination")
    print("-" * 72)

    for src, dst in moves:
        kind = "DIR " if src.is_dir() else "FILE"
        print(f"  [{kind}] {src.name:40s}  ->  {dst.parent.name}/")

    print(f"\nTotal: {len(moves)} item(s) across {len(date_folders)} date folder(s).")

    if dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")
        return

    confirm = input("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    errors = []
    for src, dst in moves:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Avoid overwriting: rename if destination already exists
            final_dst = dst
            counter = 1
            while final_dst.exists():
                final_dst = dst.parent / f"{dst.stem}_{counter}{dst.suffix}"
                counter += 1
            shutil.move(str(src), str(final_dst))
            print(f"  Moved: {src.name} -> {final_dst.parent.name}/")
        except Exception as e:
            errors.append((src, e))
            print(f"  ERROR moving {src.name}: {e}")

    print(f"\nDone. {len(moves) - len(errors)} moved, {len(errors)} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize Desktop by date.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be moved without making any changes.",
    )
    args = parser.parse_args()
    organize(dry_run=args.dry_run)

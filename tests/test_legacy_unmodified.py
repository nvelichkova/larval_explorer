"""legacy/upstream_scripts/ is read-only reference code (CLAUDE.md §6)."""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "legacy" / "upstream_scripts"
CHECKSUMS = REPO_ROOT / "tests" / "data" / "legacy_sha256.txt"


def test_legacy_files_match_recorded_checksums():
    expected = dict(
        reversed(line.split(maxsplit=1)) for line in CHECKSUMS.read_text().splitlines() if line.strip()
    )
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in LEGACY_DIR.iterdir() if path.is_file()
    }
    assert actual == expected

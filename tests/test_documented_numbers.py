"""
The setup documentation agrees with itself, and with what was measured (M7).

M7 is not really a documentation finding — it is a *consistency* finding.  The
CLAP download was stated three ways in three places:

    README.md              "requires ~1 GB model download"
    start.sh               "requires ~4 GB download on first run"
    embedding_generator.py REQUIRED_DISK_SPACE_MB = 700

All three were wrong, and the third was wrong in the direction that costs
something: the pre-flight check passed on a disk the download then filled.
Rewriting them once fixes nothing on its own — they drifted apart because
nothing held them together.  This is what holds them together.

The measurement, taken 23 July 2026 on the machine in the audit header:

    du -sb ~/.cache/huggingface/hub/models--laion--clap-htsat-unfused
    1232327859        # 1.15 GiB

It is twice the model's size because `main` carries only `pytorch_model.bin`
(614,525,833 B) and transformers 5.1.0 additionally fetches the safetensors
conversion from `refs/pr/3` (614,431,440 B).  Both stay cached, and the refs
directory shows both were pulled.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The measured cache size, in MiB.  1,232,327,859 B / 1024².
MEASURED_MODEL_CACHE_MB = 1175.2
# The measured artifact: track_embeddings.npz is 47,731,428 B.
MEASURED_ARTIFACT_MB = 45.5


def _readme():
    return (REPO_ROOT / "README.md").read_text()


def _start_sh():
    return (REPO_ROOT / "start.sh").read_text()


def test_the_preflight_check_covers_what_is_actually_downloaded():
    """
    The one that mattered: at 700 MB the check passed and the download then ran
    the disk out, which is worse than not checking, because it looks like an
    endorsement.
    """
    from embedding_generator import CLAPEmbeddingGenerator as EmbeddingGenerator

    assert EmbeddingGenerator.REQUIRED_DISK_SPACE_MB >= MEASURED_MODEL_CACHE_MB
    assert EmbeddingGenerator.MODEL_CACHE_MB >= MEASURED_MODEL_CACHE_MB
    assert EmbeddingGenerator.ARTIFACT_MB >= MEASURED_ARTIFACT_MB
    assert (EmbeddingGenerator.REQUIRED_DISK_SPACE_MB
            == EmbeddingGenerator.MODEL_CACHE_MB + EmbeddingGenerator.ARTIFACT_MB)


def test_the_check_is_not_wildly_over_stated_either():
    """
    A check that demands 10 GB refuses runs that would have worked, which is the
    same dishonesty pointing the other way.
    """
    from embedding_generator import CLAPEmbeddingGenerator as EmbeddingGenerator

    assert EmbeddingGenerator.REQUIRED_DISK_SPACE_MB < 1.5 * (
        MEASURED_MODEL_CACHE_MB + MEASURED_ARTIFACT_MB)


@pytest.mark.parametrize("stale", [
    "~1 GB model download",
    "~4 GB",
    "requires ~4 GB download",
    "1-60 min",
    "~1–60 min",
])
def test_no_stale_size_or_time_claim_survives(stale):
    """
    A comment explaining what a number replaced is prose, not a claim — the same
    exemption `test_deletions.py` makes, and for the same reason: the removals
    being legible is the point.  What must not survive is a line a *user* reads.
    """
    offenders = []
    for name, text in (("README.md", _readme()), ("start.sh", _start_sh())):
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#") and name == "start.sh":
                continue
            if stale in line:
                offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert offenders == [], f"{stale!r} is still claimed: {offenders}"


def test_the_download_size_is_stated_the_same_way_everywhere():
    """
    Both user-facing places name the measured figure.  The generator's constant
    is checked against the same measurement above, so all three now agree by
    construction rather than by coincidence.
    """
    assert "1.15 GB" in _readme()
    assert "1.15 GB" in _start_sh()


def test_a_cpu_only_estimate_exists():
    """
    CPU is the default fallback when there is no CUDA device, and the docs gave
    no figure for it at all — the one case a reader most needs a number for.
    """
    readme = _readme()
    assert re.search(r"CPU only", readme, re.I)
    assert "17.0 windows/s" in readme
    assert re.search(r"25.{0,3}35 min", readme)


def test_the_gpu_figure_is_the_one_that_was_recorded():
    """§10b: 5 min 23 s, 674 tracks, 24,494 windows, 75.8 windows/s."""
    readme = _readme()
    assert "5 min 23 s" in readme
    assert "24,494" in readme
    assert "75.8 windows/s" in readme


def test_macos_is_not_claimed_as_supported():
    """
    The README said "A terminal (Linux or macOS)" and nothing had ever been run
    there.  `start.sh`'s bash 4+ expansion — a hard syntax error on the bash 3.2
    macOS ships — was fixed in Stage 0, but album art still has no path there
    and nothing has been tested.  Either test it or drop the claim; this is the
    claim being dropped.
    """
    readme = _readme()
    assert "(Linux or macOS)" not in readme
    assert re.search(r"macOS is \*?untested", readme)


def test_start_sh_uses_no_bash_4_only_syntax():
    """
    The reason the macOS claim was not merely untested but impossible: `${VAR,,}`
    is a bash 4+ lowercase expansion and a *syntax* error on bash 3.2, so the
    script could not start at all.  Fixed in Stage 0; this keeps it fixed.
    """
    source = _start_sh()
    offenders = []
    patterns = {
        r"\$\{[A-Za-z_][A-Za-z0-9_]*,,?\}": "${VAR,,} lowercase expansion",
        r"\$\{[A-Za-z_][A-Za-z0-9_]*\^\^?\}": "${VAR^^} uppercase expansion",
        r"^\s*declare\s+-A": "associative array",
        r"^\s*(mapfile|readarray)\b": "mapfile/readarray",
    }
    for lineno, line in enumerate(source.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for pattern, what in patterns.items():
            if re.search(pattern, line):
                offenders.append(f"start.sh:{lineno}: {what}: {line.strip()}")
    assert offenders == []


def test_the_start_sh_control_banner_lists_every_binding():
    """
    E1: the launcher's control banner was a third, drifted copy of the binding
    list — it omitted `[↑↓] History` and `[ENTER] Replay`, the exact L9/M7 drift
    the rewrite prides itself on closing, while the docs claimed "the footer, the
    README table and both interfaces advertise exactly this list."
    Bind the banner to the full set here so it cannot drift again — and so a new
    binding (the `[V]` pass, G1) has to be added to the launcher too.
    """
    banner = _start_sh()
    required = [
        "Play / Pause",   # SPACE
        "Skip",           # N
        "Pass",           # V (G1)
        "Like",           # L
        "history cursor", # ↑ / ↓ (the E1 omission)
        "Zoom",           # + / − (G3)
        "Reset",          # ENTER over the cloud (G3)
        "Replay",         # ENTER over the history  (the E1 omission)
        "selected cloud track",  # P — play the clicked point now
        "Cycle panes",    # T (G3)
        "F1 / F2 / F3",   # direct pane jumps (G3)
        "mouse",          # right-drag / scroll / click / slider (G3)
        "slider",         # the orbit-speed slider (G3)
        "Volume",         # , / .
        "Seek",           # ← / →
        "Model info",     # I
        "Keybinding",     # K — the popup that replaced the footer
        "Quit",           # Q
    ]
    missing = [r for r in required if r not in banner]
    assert missing == [], f"start.sh control banner is missing: {missing}"


def test_the_pass_binding_is_advertised_on_every_user_facing_surface():
    """
    G1's binding must appear wherever the controls are listed, or the surfaces
    drift the way E1 documents.  (The footer and both interfaces are driven
    through the real key handler by the TUI tests; these are the two prose
    surfaces a test can read directly.)
    """
    assert "[V] Pass" in _readme() or "| `V` |" in _readme()
    assert "Pass" in _start_sh()


def test_the_readme_says_what_unliking_does_to_the_model():
    """
    §8's trap 1: whichever of the two designs is chosen, the README has to say
    which one it is — "un-like" that silently does nothing to the taste model
    and "un-like" that rebuilds it are different promises.
    """
    readme = _readme()
    assert re.search(r"press again.*un-like", readme, re.I)
    assert "recomputes" in readme or "recompute" in readme
    # And the exception, which the user can otherwise only discover from a
    # console line they may not be looking at.
    assert "1000 events" in readme
    assert re.search(r"display-only", readme, re.I)

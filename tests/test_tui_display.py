"""
The display, driven through the real widget tree (audit H1b/H1c/H1d/L4/L9).

Nothing in the suite constructed this tree before Stage 3.  These tests drive the
actual `AdaptiveDJTUI` against `FakeMPD` and the real selection stack, and assert
on rendered output and on FakeMPD's call log — not on the TUI's own bookkeeping,
which would prove only that it agrees with itself.
"""

import numpy as np
import pytest
import urwid

from vibe_readout import UNSEEDED_LINE


SIZE = (100, 40)


def rendered(tui, size=SIZE):
    canvas = tui.frame.render(size, focus=True)
    return [row.decode('utf-8', 'replace') for row in canvas.text]


def panel_text(tui, size=SIZE):
    return "\n".join(rendered(tui, size))


def start_playing(tui, tracks=2):
    """Put tracks in FakeMPD's queue and start it, the way a session begins."""
    mpd = tui.dj.mpd_controller
    for track in mpd.known_tracks[:tracks]:
        mpd.add_track(track)
    mpd.play()
    return mpd


def show_history(tui):
    """
    Reveal the session-history panel.

    Phase 4 (G3) made the vibe cloud the default body and moved history onto a
    toggle ([H]); the arrow keys and ENTER navigate the history only while it is
    the focused view.  Tests that assert on the history panel or its bindings
    switch to it first, the way a listener would.
    """
    from tui import BODY_HISTORY
    tui._set_body_view(BODY_HISTORY)


# ── Key routing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["up", "down", "enter"])
def test_the_new_bindings_reach_the_handler_rather_than_a_listbox(tui, key):
    """
    The console and session panels are ListBoxes, and a ListBox is selectable —
    so the body Pile focuses the first one and swallows ↑↓ before
    `unhandled_input` ever sees them.  Both panels are wrapped in
    `WidgetDisable` for exactly this reason; this asserts the consequence.

    `keypress` returning the key unchanged is urwid's "nobody handled this",
    which is what routes it to `unhandled_input`.
    """
    assert tui.frame.keypress(SIZE, key) == key


def test_the_body_is_not_selectable(tui):
    assert tui.main_pile.selectable() is False


def test_up_and_down_move_the_history_cursor(tui):
    show_history(tui)
    tui.history.note_playing("a.flac")
    tui.history.note_playing("b.flac")
    tui.history.note_playing("c.flac")

    tui._handle_input("down")
    assert tui.history.focused_track() == "b.flac"
    tui._handle_input("up")
    assert tui.history.focused_track() == "c.flac"


def test_scrolling_an_empty_history_is_harmless(tui):
    show_history(tui)
    tui._handle_input("down")
    tui._handle_input("up")
    tui._handle_input("enter")
    assert tui.history.focused_track() is None


# ── ENTER: replay a history entry ────────────────────────────────────────────


def test_enter_queues_the_focused_track_as_next(tui):
    mpd = start_playing(tui, tracks=2)
    current, lookahead = mpd.queue[0], mpd.queue[1]
    replay = mpd.known_tracks[7]

    show_history(tui)
    tui.history.note_playing(current)
    tui.history.note_playing(replay)
    tui.history.note_playing(current)
    tui.history.move_focus(+1)
    assert tui.history.focused_track() == replay

    mpd.calls.clear()
    tui._handle_input("enter")

    assert mpd.queue == [current, replay], "the lookahead was replaced"
    assert lookahead not in mpd.queue


def test_enter_adds_before_it_deletes(tui):
    """
    The ordering is the same concern as C4's.  Adding first means the queue is
    momentarily three deep and never one deep, so a refusal from MPD leaves the
    session exactly as it was — where deleting first and failing to add would
    drop the lookahead until the poller noticed.
    """
    mpd = start_playing(tui, tracks=2)
    show_history(tui)
    tui.history.note_playing(mpd.queue[0])
    mpd.calls.clear()

    tui._handle_input("enter")

    add_at = next(i for i, c in enumerate(mpd.calls) if c.startswith('add:'))
    del_at = next(i for i, c in enumerate(mpd.calls) if c.startswith('del:'))
    assert add_at < del_at
    assert 'play' not in mpd.calls, "replaying must not restart playback"
    assert 'next' not in mpd.calls, "replaying must not advance"


def test_a_refused_track_leaves_the_queue_untouched(tui):
    mpd = start_playing(tui, tracks=2)
    show_history(tui)
    before = list(mpd.queue)
    replay = mpd.known_tracks[9]
    mpd.refuse.add(replay)

    tui.history.note_playing(mpd.queue[0])
    tui.history.note_playing(replay)
    tui.history.note_playing(mpd.queue[0])
    tui.history.move_focus(+1)

    tui._handle_input("enter")
    assert mpd.queue == before


# ── The Session panel ────────────────────────────────────────────────────────


def test_the_panel_shows_the_lookahead_and_the_history(tui):
    mpd = start_playing(tui, tracks=2)
    show_history(tui)
    tui._update_display()
    text = panel_text(tui)

    assert "Session" in text, "the panel is titled Session, not Up Next"
    assert "↓ next:" in text
    assert "Test Artist" in text


def test_the_history_panel_lists_what_played_newest_first(tui):
    mpd = start_playing(tui, tracks=3)
    show_history(tui)
    tui._update_display()
    mpd.finish_track()
    tui._update_display()
    mpd.finish_track()
    tui._update_display()

    # History rows read `<marks> Test Artist – <title>`; the Now Playing panel's
    # `Artist:  Test Artist` and the `↓ next:` line are not of that shape.
    history_rows = [line for line in rendered(tui)
                    if "Test Artist – " in line and "↓ next:" not in line]
    assert len(history_rows) == 3, history_rows
    stems = [line.split("Test Artist – ")[1].strip() for line in history_rows]
    assert stems == sorted(stems, reverse=True), f"newest first: {stems}"


def test_a_skipped_track_is_marked_in_the_panel(tui):
    """
    Driven through the real `FeedbackHandler`, so the mark comes from the event
    the player actually recorded rather than from the keypress.
    """
    mpd = start_playing(tui, tracks=3)
    show_history(tui)
    tui._update_display()
    current = mpd.queue[0]

    tui.dj.feedback_handler.process_skip(current)
    tui._update_display()

    entry = next(e for e in tui.history.entries if e.track == current)
    assert entry.outcome == "⏭"
    assert "⏭" in panel_text(tui)


def test_pressing_v_passes_the_track_and_marks_it_in_the_panel(tui):
    """
    The [V] binding routes through the real `_handle_input` to `_pass_track`,
    which advances via the orchestrator and marks the passed track `»` (audit
    G1).  The orchestrator's MPD/queue advance is exercised by test_neutral_skip;
    here the wiring under test is key → history mark, so the advance is stubbed.
    """
    mpd = start_playing(tui, tracks=3)
    show_history(tui)
    tui._update_display()
    passed = mpd.queue[0]

    tui.dj.neutral_skip_current_track = lambda: passed
    tui._handle_input("v")
    tui._update_display()

    entry = next(e for e in tui.history.entries if e.track == passed)
    assert entry.outcome == "»"
    assert "»" in panel_text(tui)


def test_a_pass_that_does_nothing_marks_nothing(tui):
    """A pass the orchestrator refused (returns None) must not mark the panel."""
    start_playing(tui, tracks=3)
    tui._update_display()

    tui.dj.neutral_skip_current_track = lambda: None
    tui._handle_input("v")
    tui._update_display()

    assert all(e.outcome != "»" for e in tui.history.entries)


def test_a_completed_track_is_marked_in_the_panel(tui):
    mpd = start_playing(tui, tracks=3)
    tui._update_display()
    current = mpd.queue[0]

    tui.dj.feedback_handler.process_full_listen(current)
    tui._update_display()

    entry = next(e for e in tui.history.entries if e.track == current)
    assert entry.outcome == "✓"


def test_liking_shows_a_heart_in_the_panel_and_on_the_track_line(tui):
    mpd = start_playing(tui, tracks=2)
    show_history(tui)
    tui._update_display()

    tui._handle_input("l")
    tui._update_display()

    assert tui.history.is_liked(mpd.queue[0])
    assert "♥" in panel_text(tui)
    assert "❤" in panel_text(tui), "the Now Playing track line too"


def test_events_from_previous_sessions_do_not_back_fill_the_panel(tui):
    """
    `feedback_history` is loaded from disk and spans every session.  `run()`
    takes the cursor; here it is taken the same way so the effect is asserted.
    """
    handler = tui.dj.feedback_handler
    mpd = start_playing(tui, tracks=2)
    current = mpd.queue[0]

    # Pretend last week skipped the track that is playing tonight.
    handler.feedback_history.append({'type': 'skip', 'track': current})
    tui._feedback_cursor = len(handler.feedback_history)

    tui._update_display()
    entry = next(e for e in tui.history.entries if e.track == current)
    assert entry.outcome is None


def test_an_empty_history_says_so(tui):
    show_history(tui)
    tui._update_display()
    assert "nothing has played yet" in panel_text(tui)


def test_the_panel_asks_for_tags_per_track_not_per_playlist(tui):
    """
    Audit §8, Stage 3 trap 2: `get_playlist_metadata()` was built from
    `mpc playlist`, and with `consume on` a played track is gone from it — so the
    only existing metadata path covered exactly the tracks this panel does not
    show.  `fetch_track_tags` is per-track.
    """
    mpd = start_playing(tui, tracks=2)
    show_history(tui)
    assert not hasattr(mpd, 'get_playlist_metadata')

    tui._update_display()
    assert any(call.startswith('tags:') for call in mpd.calls)


# ── The vibe readout ─────────────────────────────────────────────────────────


def test_the_vibe_line_refuses_before_anything_has_played(tui):
    """
    The session vector starts at zero (L7) and `bank.top(zeros)` answers rather
    than refusing.  Driven through the real `SessionState`, whose `is_seeded()`
    is the gate.
    """
    assert tui.dj.session_state.is_seeded() is False
    tui._update_display()

    assert tui.descriptor_text.text == UNSEEDED_LINE
    assert "nothing has played yet" in panel_text(tui)


def test_the_vibe_line_names_descriptors_once_a_track_has_played(tui):
    mpd = start_playing(tui, tracks=2)
    tui.dj.session_state.seed(tui.dj.track_library.get_embedding(mpd.queue[0]))
    tui._update_display()

    line = tui.descriptor_text.text
    assert line.startswith("♪ ")
    assert line != UNSEEDED_LINE
    words = line[2:].split(" · ")
    assert len(words) == 3
    assert all(word in tui.dj.descriptor_bank.labels for word in words)
    assert line in panel_text(tui)


def test_the_drift_line_carries_the_count_and_the_track_total(tui):
    mpd = start_playing(tui, tracks=2)
    library = tui.dj.track_library
    session = tui.dj.session_state

    session.seed(library.get_embedding(mpd.known_tracks[0]))
    tui._update_display()
    for track in mpd.known_tracks[1:7]:
        session.update(library.get_embedding(track))
        tui._update_display()

    line = tui.drift_text.text
    assert line.startswith("⟳ ")
    assert "held over" in line
    assert f"{session.tracks_played} played" in line


def test_a_missing_descriptor_bank_is_survivable(dj_stub, fake_art):
    """The bank is a display feature; the player must still play without it."""
    import signal as _signal

    from tui import AdaptiveDJTUI

    dj_stub.descriptor_bank = None
    previous = _signal.getsignal(_signal.SIGWINCH)
    instance = AdaptiveDJTUI(dj_stub, art_renderer=fake_art)
    try:
        instance._update_display()
        assert "no descriptor bank" in instance.descriptor_text.text
        instance.frame.render(SIZE, focus=True)
    finally:
        _signal.signal(_signal.SIGWINCH, previous)


# ── The [I] inspector ────────────────────────────────────────────────────────


def test_the_inspector_reports_the_descriptors_for_both_vectors(tui):
    mpd = start_playing(tui, tracks=2)
    library = tui.dj.track_library
    tui.dj.session_state.seed(library.get_embedding(mpd.queue[0]))
    tui.dj.user_taste.update_from_like(library.get_embedding(mpd.queue[0]))

    lines = tui._model_info_lines()
    text = "\n".join(lines)

    assert "DESCRIPTORS" in text
    session_row = next(line for line in lines if line.strip().startswith("session"))
    taste_row = next(line for line in lines if line.strip().startswith("taste "))
    assert any(label in session_row for label in tui.dj.descriptor_bank.labels)
    assert any(label in taste_row for label in tui.dj.descriptor_bank.labels)


def test_the_inspector_refuses_the_descriptors_it_cannot_justify(tui):
    """
    Both vectors are unseeded at startup, and z-scoring either would produce a
    confident-looking ranking of the bank's own baselines (H1).
    """
    lines = tui._model_info_lines()
    session_row = next(line for line in lines if line.strip().startswith("session"))
    taste_row = next(line for line in lines if line.strip().startswith("taste "))

    assert "nothing has played yet" in session_row
    assert "no positive signal yet" in taste_row


def test_the_inspector_still_reports_the_stage_two_machinery(tui):
    """Stage 3 adds rows; it must not drop the ones Stage 2 earned."""
    text = "\n".join(tui._model_info_lines())
    for expected in ("rank-Boltzmann", "β earned", "next skip targets",
                     "CURRENT SCORING WEIGHTS", "τ"):
        assert expected in text


def test_the_inspector_scrolls_rather_than_hiding_its_tail(tui, monkeypatch):
    """
    Stage 2 sized this overlay to its content after the sampling rows overflowed
    a fixed height and the last third was silently cut off.  The descriptor rows
    push it past a 40-row terminal, so it is a ListBox now — an inspector that
    hides part of what it is inspecting is worse than none.
    """
    keys = [["down"], ["down"], ["q"]]
    drawn = []

    monkeypatch.setattr(tui.loop, 'draw_screen',
                        lambda: drawn.append(tui.loop.widget))
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (100, 20))
    monkeypatch.setattr(tui.loop.screen, 'get_input', lambda: keys.pop(0))

    original = tui.loop.widget
    lines = tui._show_model_info()

    assert len(drawn) == 3, "two scrolls then a dismissal"
    assert isinstance(drawn[0], urwid.Overlay)
    assert tui.loop.widget is original, "the overlay is dismissed"

    # The overlay is taller than the 20-row screen allows, so it must scroll.
    assert len(lines) > 18


def test_the_inspector_is_dismissed_by_any_non_scroll_key(tui, monkeypatch):
    monkeypatch.setattr(tui.loop, 'draw_screen', lambda: None)
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (100, 40))
    monkeypatch.setattr(tui.loop.screen, 'get_input', lambda: ['x'])

    original = tui.loop.widget
    tui._show_model_info()
    assert tui.loop.widget is original


# ── Album art ────────────────────────────────────────────────────────────────


def test_the_art_is_rendered_at_the_derived_geometry(tui, fake_art, monkeypatch,
                                                     tmp_path):
    art = tmp_path / "cover.jpg"
    art.write_bytes(b"not really a jpeg")

    mpd = start_playing(tui, tracks=2)
    fake_art.art_for[mpd.queue[0]] = art
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: SIZE)

    tui._update_display()

    assert fake_art.renders, "art was never rendered"
    call = fake_art.renders[-1]
    x, y, width, height = tui._art_geometry(*SIZE)
    assert (call['x'], call['y'], call['width'], call['height']) == \
        (x, y, width, height)


def test_a_terminal_too_short_for_art_clears_instead_of_smearing(tui, fake_art,
                                                                monkeypatch,
                                                                tmp_path):
    art = tmp_path / "cover.jpg"
    art.write_bytes(b"not really a jpeg")

    mpd = start_playing(tui, tracks=2)
    fake_art.art_for[mpd.queue[0]] = art
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (100, 8))

    tui._update_display()

    assert not fake_art.renders
    assert fake_art.clears >= 1


# ── L9: the bindings agree with each other ───────────────────────────────────


def test_the_footer_advertises_exactly_the_keys_that_are_bound(tui):
    """
    The footer, the README table and the fallback-mode bindings disagreed three
    ways (audit L9).  A footer that advertises a key doing nothing is the same
    dishonesty the rest of this rewrite removes — and so is a key that works but
    is not advertised, which is what ↑↓ and ENTER were.
    """
    footer = tui.frame.footer.base_widget.text
    for advertised in ("[SPACE]", "[N]", "[V]", "[L]", "[↑↓]", "[ENTER]",
                       "[+−]", "[B]", "[T]", "[,.]", "[←→]", "[I]", "[Q]"):
        assert advertised in footer

    # Every advertised key reaches a distinct dispatch target.  Recorded through
    # the real `_handle_input`, so a binding removed from the table fails here
    # rather than quietly leaving the footer lying.  The pane cycle is `[T]`;
    # F1/F2/F3 jump straight to a pane but are a deliberately unadvertised
    # convenience, so they are not in this footer list.  ENTER routes to a single
    # method whose *effect* follows the view (reset vs replay) — that branch has
    # its own test; here the invariant is the fixed key → method table (§4).
    fired = []
    for name in ('_toggle_play_pause', '_skip_track', '_pass_track',
                 '_like_track', '_arrow_up', '_arrow_down', '_zoom_in',
                 '_zoom_out', '_enter_action', '_cycle_scaffold',
                 '_toggle_pane', '_volume_down',
                 '_volume_up', '_seek_backward', '_seek_forward',
                 '_show_model_info', '_quit'):
        monkeypatched = (lambda n: lambda *a, **k: fired.append(n))(name)
        setattr(tui, name, monkeypatched)

    for key in (" ", "n", "v", "l", "up", "down", "+", "-", "enter", "b", "t",
                ",", ".", "left", "right", "i", "q"):
        before = len(fired)
        tui._handle_input(key)
        assert len(fired) == before + 1, f"[{key}] is advertised but does nothing"

    assert fired == ['_toggle_play_pause', '_skip_track', '_pass_track',
                     '_like_track', '_arrow_up', '_arrow_down', '_zoom_in',
                     '_zoom_out', '_enter_action', '_cycle_scaffold',
                     '_toggle_pane', '_volume_down',
                     '_volume_up', '_seek_backward', '_seek_forward',
                     '_show_model_info', '_quit']


def test_the_footer_wraps_rather_than_truncating_on_a_narrow_terminal(tui):
    """It is a flow widget now, so a narrow terminal gets two rows of hints
    instead of a silently clipped one."""
    assert tui.frame.footer.rows((60,)) > 1
    lines = rendered(tui, (60, 30))
    assert "[Q] Quit" in "".join(lines[-2:])


def test_the_palette_defines_exactly_the_attributes_the_widgets_use(tui):
    """
    A palette entry with no widget is dead weight (D7); a widget attribute with
    no palette entry renders in the terminal default and looks like a styling
    bug.  The queue panel's two entries outlived the panel itself.
    """
    import re
    from pathlib import Path

    source = Path(tui.__module__.replace('.', '/') + '.py')
    if not source.exists():
        source = Path(__file__).resolve().parent.parent / "src" / "tui.py"
    text = source.read_text()

    palette_block = re.search(r"palette = \[(.*?)\n        \]", text, re.S).group(1)
    defined = set(re.findall(r'\("([a-z_]+)"', palette_block))
    used = set(re.findall(r'"([a-z_]+)"', text.replace(palette_block, "")))

    assert defined <= used, f"palette entries no widget uses: {defined - used}"


def test_the_session_keeps_being_observed_while_the_inspector_is_open(tui,
                                                                     monkeypatch):
    """
    The `[I]` loop owns the thread that records history, and ↑↓ no longer
    dismiss the overlay — so it can now be held open across a whole track.

    Measured before the fix: a track that started *and* finished while `[I]` was
    up never reached the history panel, and its `✓` was dropped, because
    `apply_event` had no entry to attach it to.  The panel's entire claim is that
    it lists what actually played.  Stage 2's inspector closed on any key, so it
    could not be held open long enough for this to bite.
    """
    mpd = start_playing(tui, tracks=5)
    tui._update_display()
    first, second = mpd.known_tracks[0], mpd.known_tracks[1]

    # `[]` is a timeout wake-up with no keypress — the case that matters.
    keys = [[], ["down"], [], ["q"]]
    monkeypatch.setattr(tui.loop, 'draw_screen', lambda: None)
    monkeypatch.setattr(tui.loop.screen, 'get_cols_rows', lambda: (100, 20))

    def fake_input():
        if len(keys) == 4:
            tui.dj.feedback_handler.process_full_listen(first)
            mpd.finish_track()
        if len(keys) == 3:
            tui.dj.feedback_handler.process_full_listen(second)
            mpd.finish_track()
        return keys.pop(0)

    monkeypatch.setattr(tui.loop.screen, 'get_input', fake_input)
    tui._show_model_info()
    tui._update_display()

    played = {e.track: e.outcome for e in tui.history.entries}
    assert first in played and second in played, (
        "a track that played entirely while [I] was open is missing from history")
    assert played[first] == "✓" and played[second] == "✓"

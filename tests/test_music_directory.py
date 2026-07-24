"""
Finding MPD's music directory (audit M3).

The old default was `/var/lib/mpd/music`, which worked on the development
machine only because that path happened to be a symlink to the real library.  A
wrong value costs an entire embedding-generation run, so detection is read from
MPD's own config and the answer is proved against real track paths.
"""

import os
from pathlib import Path

import pytest

import music_directory as md


# ------------------------------------------------------------------- parsing

@pytest.mark.parametrize("line,expected", [
    ('music_directory "/mnt/storage/music"', '/mnt/storage/music'),
    ('music_directory /mnt/storage/music', '/mnt/storage/music'),
    ('   music_directory   "/mnt/with space/music"  ', '/mnt/with space/music'),
    ('music_directory "/mnt/storage/music"   # the good stuff', '/mnt/storage/music'),
])
def test_directive_forms_all_parse(line, expected):
    assert md.parse_music_directory(line) == expected


def test_a_hash_inside_a_quoted_value_is_kept(tmp_path):
    """
    E3: MPD treats `#` as literal inside quotes, so a directory whose name
    contains one must not be truncated at it.  The old `[^"#]` value class cut
    it off.
    """
    assert md.parse_music_directory(
        'music_directory "/mnt/music/#rare/set"') == '/mnt/music/#rare/set'


def test_an_unquoted_value_still_stops_at_a_trailing_comment():
    assert md.parse_music_directory(
        'music_directory /mnt/music/set   # the good stuff') == '/mnt/music/set'


def test_commented_out_directives_are_ignored():
    text = '# music_directory "/wrong/one"\nmusic_directory "/right/one"\n'
    assert md.parse_music_directory(text) == '/right/one'


def test_first_directive_wins():
    text = 'music_directory "/first"\nmusic_directory "/second"\n'
    assert md.parse_music_directory(text) == '/first'


def test_absent_directive_returns_none():
    assert md.parse_music_directory('bind_to_address "localhost"\nport "6600"\n') is None


# ----------------------------------------------------------------- detection

def test_env_var_beats_every_config_file(tmp_path, monkeypatch):
    conf = tmp_path / 'mpd.conf'
    conf.write_text('music_directory "/from/config"\n')
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(conf),))
    monkeypatch.setenv('MPD_MUSIC_DIR', '/from/env')

    path, source = md.detect_music_directory()
    assert path == Path('/from/env')
    assert source == 'MPD_MUSIC_DIR'


def test_config_file_is_read_when_env_is_unset(tmp_path, monkeypatch):
    conf = tmp_path / 'mpd.conf'
    conf.write_text('port "6600"\nmusic_directory "~/Music"\n')
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(conf),))
    monkeypatch.delenv('MPD_MUSIC_DIR', raising=False)

    path, source = md.detect_music_directory()
    assert path == Path.home() / 'Music'      # ~ expanded
    assert source == str(conf)


def test_a_config_value_dollar_var_is_taken_literally(tmp_path, monkeypatch):
    """
    E3: MPD does not expand `$VAR` in its config (only a leading `~`), so neither
    do we — expanding it would rewrite a `$` MPD would have kept.
    """
    conf = tmp_path / 'mpd.conf'
    conf.write_text('music_directory "/mnt/music/$MYVAR/x"\n')
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(conf),))
    monkeypatch.delenv('MPD_MUSIC_DIR', raising=False)
    monkeypatch.setenv('MYVAR', 'SHOULD_NOT_APPEAR')

    path, _ = md.detect_music_directory()
    assert path == Path('/mnt/music/$MYVAR/x')


def test_our_own_env_override_still_expands_shell_style(monkeypatch):
    """The MPD_MUSIC_DIR knob is ours, not MPD's, so shell-style expansion of it
    is what a user setting it would expect."""
    monkeypatch.setenv('MPD_MUSIC_DIR', '$HOME/Music')
    path, source = md.detect_music_directory()
    assert path == Path(os.path.expandvars('$HOME/Music'))
    assert source == 'MPD_MUSIC_DIR'


def test_earlier_candidates_win(tmp_path, monkeypatch):
    """The user's own config outranks the system one, as it does for MPD."""
    user = tmp_path / 'user.conf'
    system = tmp_path / 'system.conf'
    user.write_text('music_directory "/user/music"\n')
    system.write_text('music_directory "/system/music"\n')
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(user), str(system)))
    monkeypatch.delenv('MPD_MUSIC_DIR', raising=False)

    assert md.detect_music_directory()[0] == Path('/user/music')


def test_missing_files_are_skipped_not_fatal(tmp_path, monkeypatch):
    real = tmp_path / 'real.conf'
    real.write_text('music_directory "/real/music"\n')
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(tmp_path / 'nope.conf'), str(real)))
    monkeypatch.delenv('MPD_MUSIC_DIR', raising=False)

    assert md.detect_music_directory()[0] == Path('/real/music')


def test_nothing_found_returns_none_rather_than_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(md, 'CONFIG_CANDIDATES', (str(tmp_path / 'absent.conf'),))
    monkeypatch.delenv('MPD_MUSIC_DIR', raising=False)

    path, source = md.detect_music_directory()
    assert path is None and source == 'not found'


# ---------------------------------------------------------------- validation

def test_validation_passes_when_tracks_resolve(tmp_path):
    (tmp_path / 'a').mkdir()
    for name in ('a/one.flac', 'a/two.flac'):
        (tmp_path / name).write_bytes(b'')

    ok, message = md.validate_music_directory(tmp_path, ['a/one.flac', 'a/two.flac'])
    assert ok, message


def test_validation_fails_on_the_wrong_directory(tmp_path):
    ok, message = md.validate_music_directory(tmp_path, ['a/one.flac'])
    assert not ok
    assert 'MPD_MUSIC_DIR' in message


def test_validation_fails_when_the_directory_does_not_exist(tmp_path):
    ok, message = md.validate_music_directory(tmp_path / 'gone', ['a.flac'])
    assert not ok and 'does not exist' in message


def test_validation_fails_on_none_rather_than_crashing():
    ok, message = md.validate_music_directory(None, ['a.flac'])
    assert not ok and 'could not be determined' in message


def test_validation_reports_an_empty_mpd_database(tmp_path):
    ok, message = md.validate_music_directory(tmp_path, [])
    assert not ok and 'mpc update' in message


def test_probes_are_spread_across_the_library():
    """
    One probe is not enough: any single file can be missing while the directory
    is right, and a partially-mounted library only shows up if the samples are
    spread.
    """
    tracks = [f"{i:03d}.flac" for i in range(100)]
    probes = md.sample_tracks(tracks, count=5)

    assert len(probes) == 5
    assert len(set(probes)) == 5
    assert probes[0] == '000.flac'
    assert probes[-1] != probes[0]


def test_sampling_a_short_library_returns_all_of_it():
    assert md.sample_tracks(['a.flac', 'b.flac'], count=5) == ['a.flac', 'b.flac']
    assert md.sample_tracks([], count=5) == []


def test_a_partially_mounted_library_is_caught(tmp_path):
    """Half the tracks present is a failure, not a pass — the probes must not
    all land in the half that happens to exist."""
    tracks = [f"{i:03d}.flac" for i in range(100)]
    for name in tracks[:50]:
        (tmp_path / name).write_bytes(b'')

    ok, message = md.validate_music_directory(tmp_path, tracks)
    assert not ok
    assert 'probes missing' in message

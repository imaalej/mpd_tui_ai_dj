"""
The CLAP descriptor bank (audit D5 / H1).

H1's original defect was a mood word pinned to one value by thresholds chosen
for a different scale.  The replacement can fail the same way for a different
reason — the **modality gap**: CLAP's audio and text towers do not share a cone,
so raw `audio · text` dot products are not comparable across descriptors and a
naive top-3 returns the same three words forever.

Two defences are tested here, because they are the whole design:
per-descriptor z-scoring against the library's own distribution, and a build-time
variance gate that drops words carrying no information about *this* collection.
"""

import numpy as np
import pytest

import descriptor_bank as db


@pytest.fixture
def bank_inputs(rng):
    """A small synthetic library and a descriptor set with known behaviour."""
    dimension = 32
    library = rng.standard_normal((200, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    return library, dimension


def _bank_from(text_embeddings, labels, library, **kwargs):
    payload = db.build_bank(
        text_embeddings=text_embeddings,
        labels=labels,
        prompts=[f"prompt for {label}" for label in labels],
        library_centred=library,
        **kwargs,
    )
    report = payload.pop('report')
    return payload, report


# ----------------------------------------------------------------- the words

def test_the_bank_spans_complementary_axes():
    """Three words from one axis repeat themselves; three from three axes
    describe a track.  The axis grouping is the design, so it is asserted."""
    assert len(db.DESCRIPTORS) >= 45
    assert len(set(db.DESCRIPTORS)) == len(db.DESCRIPTORS), "duplicate descriptor"
    assert set(db.DESCRIPTOR_AXES) == {
        'energy', 'affect', 'texture', 'rhythm', 'setting', 'instrumentation'
    }
    for axis, labels in db.DESCRIPTOR_AXES.items():
        assert len(labels) >= 6, axis


def test_prompts_render_through_the_template():
    prompts = db.render_prompts(['hypnotic'], db.PROMPT_TEMPLATES['recording'])
    assert prompts == ['This is a recording of hypnotic music.']


def test_every_template_consumes_the_label():
    for name, template in db.PROMPT_TEMPLATES.items():
        assert 'hypnotic' in template.format('hypnotic'), name


# ---------------------------------------------------------- the variance gate

@pytest.fixture
def uneven_library(rng):
    """
    A library with a lot of variance along one axis and almost none along
    another — so a descriptor pointing at each has a genuinely different std,
    which is what the gate is supposed to notice.
    """
    dimension = 16
    library = rng.standard_normal((300, dimension)).astype(np.float32) * 0.02
    library[:, 0] = rng.standard_normal(300) * 1.0      # discriminating axis
    library[:, 1] = 1.0                                 # every track identical here
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    return library, dimension


def test_a_descriptor_that_says_the_same_thing_about_everything_is_dropped(uneven_library):
    """
    The anti-trap step (D5b).  A word whose library std is ~0 cannot be
    informative — and dividing by that std would turn noise into a confident
    z-score.
    """
    library, dimension = uneven_library
    discriminating = np.eye(1, dimension, k=0, dtype=np.float32)
    flat = np.eye(1, dimension, k=1, dtype=np.float32)
    text = np.vstack([discriminating, flat])

    _, report = _bank_from(text, ['sharp', 'flat'], library)

    assert [label for label, _ in report['dropped']] == ['flat']
    assert [label for label, _ in report['kept']] == ['sharp']


def test_the_gate_floor_is_relative_to_the_library_not_absolute(bank_inputs):
    """
    Scaling every similarity must not change which descriptors survive.  An
    absolute floor would be the same mistake C5 documents — a constant tuned
    against a scale that moves with the model, the template and the collection.
    """
    library, dimension = bank_inputs
    text = np.eye(6, dimension, dtype=np.float32)
    labels = list('abcdef')

    _, report_small = _bank_from(text, labels, library * 0.5)
    _, report_large = _bank_from(text, labels, library * 2.0)

    assert ([l for l, _ in report_small['kept']] == [l for l, _ in report_large['kept']])


def test_nothing_is_dropped_when_every_descriptor_discriminates(bank_inputs):
    library, dimension = bank_inputs
    text = np.eye(8, dimension, dtype=np.float32)
    _, report = _bank_from(text, list('abcdefgh'), library)
    assert report['dropped'] == []


def test_a_degenerate_library_keeps_the_whole_bank_rather_than_shipping_nothing(rng):
    """One track, or a library of identical vectors: every std is 0.  An empty
    bank is a worse artifact than an unfiltered one."""
    dimension = 16
    identical = np.tile(np.eye(1, dimension, dtype=np.float32), (20, 1))
    text = rng.standard_normal((5, dimension)).astype(np.float32)
    payload, report = _bank_from(text, list('abcde'), identical)
    assert len(payload['labels']) == 5
    assert report['dropped'] == []


def test_only_surviving_descriptors_reach_the_artifact(uneven_library):
    """A dropped descriptor must leave no trace — a stale mean or std would
    reintroduce the word by index alone."""
    library, dimension = uneven_library
    text = np.vstack([
        np.eye(1, dimension, k=0, dtype=np.float32),   # discriminating
        np.eye(1, dimension, k=1, dtype=np.float32),   # flat
    ])
    payload, report = _bank_from(text, ['sharp', 'flat'], library)

    assert list(payload['labels']) == ['sharp']
    assert list(payload['prompts']) == ['prompt for sharp']
    assert len(payload['mean']) == 1
    assert len(payload['std']) == 1
    assert payload['text_embeddings'].shape[0] == 1
    assert len(report['dropped']) == 1


# --------------------------------------------------------------- the z-scores

def test_z_scoring_neutralises_the_modality_gap(rng):
    """
    The failure this design exists to prevent: one descriptor sitting at a much
    higher raw similarity than another, so it wins every top-3 forever
    regardless of the music.
    """
    dimension = 24
    library = rng.standard_normal((300, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)

    loud = np.zeros(dimension, dtype=np.float32)
    loud[0] = 1.0
    quiet = np.zeros(dimension, dtype=np.float32)
    quiet[1] = 1.0
    # Give `loud` a large constant head start, exactly as the gap does.
    library = library + 0.9 * loud
    library /= np.linalg.norm(library, axis=1, keepdims=True)

    payload, _ = _bank_from(np.vstack([loud, quiet]), ['loud', 'quiet'], library)
    bank = db.DescriptorBank(payload['labels'], payload['text_embeddings'],
                             payload['mean'], payload['std'])

    raw_wins = sum(
        (bank.text_embeddings @ v).argmax() == 0 for v in library
    )
    z_wins = sum(bank.z_scores(v).argmax() == 0 for v in library)

    assert raw_wins > 0.95 * len(library), "the fixture must reproduce the gap"
    assert 0.3 * len(library) < z_wins < 0.7 * len(library), (
        f"z-scoring should split the library, not pin one word ({z_wins}/{len(library)})"
    )


def test_a_typical_track_scores_near_zero_and_an_extreme_one_high(rng):
    """z is 'unusually X for this library' — so the median track is not
    unusual, by construction."""
    dimension = 16
    library = rng.standard_normal((400, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    text = np.eye(4, dimension, dtype=np.float32)

    payload, _ = _bank_from(text, list('abcd'), library)
    bank = db.DescriptorBank(payload['labels'], payload['text_embeddings'],
                             payload['mean'], payload['std'])

    z_all = np.array([bank.z_scores(v) for v in library])
    assert abs(z_all.mean()) < 0.05
    assert abs(z_all.std() - 1.0) < 0.05

    aligned = np.zeros(dimension, dtype=np.float32)
    aligned[0] = 1.0
    assert bank.z_scores(aligned)[0] > 2.0


def test_top_returns_descending_z_scores(rng):
    dimension = 12
    library = rng.standard_normal((100, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    payload, _ = _bank_from(np.eye(6, dimension, dtype=np.float32), list('abcdef'), library)
    bank = db.DescriptorBank(payload['labels'], payload['text_embeddings'],
                             payload['mean'], payload['std'])

    top = bank.top(library[0], n=3)
    assert len(top) == 3
    assert [z for _, z in top] == sorted((z for _, z in top), reverse=True)
    assert set(label for label, _ in top) <= set(bank.labels)


def test_scoring_does_not_care_about_the_query_magnitude(rng):
    dimension = 12
    library = rng.standard_normal((100, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    payload, _ = _bank_from(np.eye(4, dimension, dtype=np.float32), list('abcd'), library)
    bank = db.DescriptorBank(payload['labels'], payload['text_embeddings'],
                             payload['mean'], payload['std'])

    assert np.allclose(bank.z_scores(library[7]), bank.z_scores(library[7] * 7.5), atol=1e-5)


# -------------------------------------------------------------- the artifact

def test_the_bank_round_trips_through_the_npz(tmp_path, rng):
    dimension = 12
    library = rng.standard_normal((80, dimension)).astype(np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    payload, _ = _bank_from(np.eye(5, dimension, dtype=np.float32), list('abcde'), library)

    path = tmp_path / 'descriptors.npz'
    np.savez_compressed(path, **payload)

    loaded = db.DescriptorBank.load(path)
    assert loaded is not None
    assert loaded.labels == list(payload['labels'])
    assert np.array_equal(loaded.mean, payload['mean'])
    assert np.allclose(loaded.z_scores(library[3]),
                       db.DescriptorBank(payload['labels'], payload['text_embeddings'],
                                         payload['mean'], payload['std']).z_scores(library[3]))


def test_an_absent_bank_loads_as_none_rather_than_raising(tmp_path):
    assert db.DescriptorBank.load(tmp_path / 'nope.npz') is None


def test_an_old_schema_is_refused(tmp_path, rng):
    path = tmp_path / 'descriptors.npz'
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        labels=np.array(['a'], dtype=np.str_),
        prompts=np.array(['a'], dtype=np.str_),
        text_embeddings=rng.standard_normal((1, 4)).astype(np.float32),
        mean=np.zeros(1, np.float32),
        std=np.ones(1, np.float32),
    )
    assert db.DescriptorBank.load(path) is None


# ------------------------------------------------------------ template choice

def test_separation_metrics_reward_independent_descriptors(rng):
    """
    A bank whose 50 words rank the library identically measures one thing, not
    fifty.  Effective rank is what makes that visible when choosing a template.
    """
    scores = rng.standard_normal((200, 10))
    independent = db.separation_metrics(scores)

    one_axis = np.tile(scores[:, :1], (1, 10)) + rng.standard_normal((200, 10)) * 0.01
    duplicated = db.separation_metrics(one_axis)

    assert independent['effective_rank'] > 8
    assert duplicated['effective_rank'] < 2


def test_separation_metrics_survive_a_constant_descriptor(rng):
    scores = rng.standard_normal((50, 4))
    scores[:, 2] = 1.0
    metrics = db.separation_metrics(scores)
    assert np.isfinite(metrics['effective_rank'])
    assert metrics['min_std'] == 0.0

from data.sampler import DefaultSampler


def test_default_sampler_returns_indices():
    sampler = DefaultSampler([0, 1, 2, 3, 4])
    assert list(sampler) == [[0], [1], [2], [3], [4]]


def test_default_sampler_len():
    sampler = DefaultSampler([0, 1, 2, 3, 4])
    assert len(sampler) == 5


def test_default_sampler_shuffle():
    sampler = DefaultSampler([0, 1, 2, 3, 4], shuffle=True)
    assert sorted(sampler) == [[0], [1], [2], [3], [4]]


def test_default_sampler_batch_size():
    sampler = DefaultSampler(
        [0, 1, 2, 3, 4, 5],
        shuffle=True,
        batch_size=2,
    )

    sampler2 = DefaultSampler(
        [0, 1, 2, 3, 4, 5],
        shuffle=True,
        batch_size=3,
    )

    sampler3 = DefaultSampler(
        [0, 1, 2, 3, 4, 5],
        shuffle=True,
        batch_size=4,
    )

    assert len(sampler) == 3
    assert len(sampler2) == 2
    assert len(sampler3) == 2
    assert len(next(iter(sampler3))) == 4

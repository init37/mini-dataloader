import pytest

from mini_dataloader.data.sampler import BatchSampler, DefaultSampler


# Fixture for creating a DefaultSampler instance
@pytest.fixture
def create_default_sampler():
    indices = list(range(10))
    return DefaultSampler(indices, shuffle=False, batch_size=3)


# Test cases for BatchSampler abstract class
def test_batch_sampler_abstract_methods():
    class TestBatchSampler(BatchSampler):
        def __iter__(self):
            yield from []

        def __len__(self):
            return 0

    test_sampler = TestBatchSampler()
    iterator = iter(test_sampler)
    batches = list(iterator)
    assert len(batches) == 0
    assert len(test_sampler) == 0


# Test cases for DefaultSampler class
def test_default_sampler_init(create_default_sampler):
    sampler = create_default_sampler
    assert sampler.indices == list(range(10))
    assert not sampler.shuffle
    assert sampler.batch_size == 3


def test_default_sampler_len(create_default_sampler):
    sampler = create_default_sampler
    assert len(sampler) == 4  # (10 + 3 - 1) // 3


def test_default_sampler_iter(create_default_sampler):
    sampler = create_default_sampler
    iterator = iter(sampler)
    batches = list(iterator)
    assert len(batches) == 4
    assert all(len(batch) == 3 for batch in batches[:-1])


def test_default_sampler_shuffle(create_default_sampler):
    sampler = DefaultSampler(list(range(10)), shuffle=True, batch_size=3)
    iterator = iter(sampler)
    batches = list(iterator)
    assert len(batches) == 4
    assert all(len(batch) == 3 for batch in batches[:-1])
    assert batches != list(range(0, 10, 3))


def test_default_sampler_invalid_batch_size():
    with pytest.raises(ValueError):
        DefaultSampler(list(range(10)), shuffle=False, batch_size=-1)

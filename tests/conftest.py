import pytest


@pytest.fixture(scope="session")
def output_directory(tmpdir_factory):

    directory = tmpdir_factory.mktemp("output")

    return directory


@pytest.fixture(scope="session")
def random_seed():

    seed = 42

    return seed

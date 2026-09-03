import numpy as np
import pytest

from hamlet.systems import InhomogeneousHeisenbergChain


def test_chain_length_is_derived_from_bonds():
    chain = InhomogeneousHeisenbergChain([3.1, 3.2, 3.3])
    assert chain.n_sites == 4
    assert chain.n_bonds == 3


def test_sampling_is_reproducible():
    a = InhomogeneousHeisenbergChain.sample(6, (3.0, 4.0), np.random.default_rng(7))
    b = InhomogeneousHeisenbergChain.sample(6, (3.0, 4.0), np.random.default_rng(7))
    np.testing.assert_allclose(a.couplings_mev, b.couplings_mev)


def test_invalid_chain_is_rejected():
    with pytest.raises(ValueError):
        InhomogeneousHeisenbergChain([])


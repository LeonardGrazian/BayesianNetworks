
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork
from utils import fuzzy_match

# constants
N_SAMPLE = 10000
FUZZY_MATCH_ERROR = 0.0001


def test_marginals_brute_force_cascade(rng):
    print('--- TESTING CASCADE ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pb0_c1 = rng.uniform()
    pb1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1], rng=rng)
    c = BinaryNode('c', [b], [pb0_c1, pb1_c1], rng=rng)
    bnet = BinaryBayesianNetwork([a, b, c])

    a_marginal_true = pa1
    b_marginal_true = pa0_b1 * (1 - pa1) + pa1_b1 * pa1
    c_marginal_true = pb0_c1 * (1 - b_marginal_true) + pb1_c1 * b_marginal_true
    ab_marginal_true = [
        (1 - pa1) * (1 - pa0_b1),
        pa1 * (1 - pa1_b1),
        (1 - pa1) * pa0_b1,
        pa1 * pa1_b1
    ]
    bc_marginal_true = [
        (1 - pa1) * (1 - pa0_b1) * (1 - pb0_c1)
            + pa1 * (1 - pa1_b1) * (1 - pb0_c1),
        (1 - pa1) * pa0_b1 * (1 - pb1_c1) + pa1 * pa1_b1 * (1 - pb1_c1),
        (1 - pa1) * (1 - pa0_b1) * pb0_c1 + pa1 * (1 - pa1_b1) * pb0_c1,
        (1 - pa1) * pa0_b1 * pb1_c1 + pa1 * pa1_b1 * pb1_c1
    ]
    ac_marginal_true = [
        (1 - pa1) * (1 - pa0_b1) * (1 - pb0_c1)
            + (1 - pa1) * pa0_b1 * (1 - pb1_c1),
        pa1 * (1 - pa1_b1) * (1 - pb0_c1) + pa1 * pa1_b1 * (1 - pb1_c1),
        (1 - pa1) * (1 - pa0_b1) * pb0_c1 + (1 - pa1) * pa0_b1 * pb1_c1,
        pa1 * (1 - pa1_b1) * pb0_c1 + pa1 * pa1_b1 * pb1_c1
    ]
    a_marginal_obs = bnet.get_marginal_by_brute_force([a])[1]
    b_marginal_obs = bnet.get_marginal_by_brute_force([b])[1]
    c_marginal_obs = bnet.get_marginal_by_brute_force([c])[1]
    ab_marginal_obs = bnet.get_marginal_by_brute_force([a, b])
    bc_marginal_obs = bnet.get_marginal_by_brute_force([b, c])
    ac_marginal_obs = bnet.get_marginal_by_brute_force([a, c])

    assert fuzzy_match(a_marginal_true, a_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(b_marginal_true, b_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(c_marginal_true, c_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ab_marginal_true, ab_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(bc_marginal_true, bc_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ac_marginal_true, ac_marginal_obs, FUZZY_MATCH_ERROR)

    print('predicted marginal is {}'.format(c_marginal_obs))
    s = 0
    for _ in range(N_SAMPLE):
        s += bnet.sample()[c]
    print('estimated marginal is {}'.format(s * 1.0 / N_SAMPLE))
    print()


def test_marginals_brute_force_common_parent(rng):
    print('--- TESTING COMMON_PARENT ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pa0_c1 = rng.uniform()
    pa1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1], rng=rng)
    c = BinaryNode('c', [a], [pa0_c1, pa1_c1], rng=rng)
    bnet = BinaryBayesianNetwork([a, b, c])

    a_marginal_true = pa1
    b_marginal_true = (1 - pa1) * pa0_b1 + pa1 * pa1_b1
    c_marginal_true = (1 - pa1) * pa0_c1 + pa1 * pa1_c1
    ab_marginal_true = [
        (1 - pa1) * (1 - pa0_b1),
        pa1 * (1 - pa1_b1),
        (1 - pa1) * pa0_b1,
        pa1 * pa1_b1
    ]
    bc_marginal_true = [
        (1 - pa1) * (1 - pa0_b1) * (1 - pa0_c1)
            + pa1 * (1 - pa1_b1) * (1 - pa1_c1),
        (1 - pa1) * pa0_b1 * (1 - pa0_c1) + pa1 * pa1_b1 * (1 - pa1_c1),
        (1 - pa1) * (1 - pa0_b1) * pa0_c1 + pa1 * (1 - pa1_b1) * pa1_c1,
        (1 - pa1) * pa0_b1 * pa0_c1 + pa1 * pa1_b1 * pa1_c1
    ]
    ac_marginal_true = [
        (1 - pa1) * (1 - pa0_c1),
        pa1 * (1 - pa1_c1),
        (1 - pa1) * pa0_c1,
        pa1 * pa1_c1
    ]
    a_marginal_obs = bnet.get_marginal_by_brute_force([a])[1]
    b_marginal_obs = bnet.get_marginal_by_brute_force([b])[1]
    c_marginal_obs = bnet.get_marginal_by_brute_force([c])[1]
    ab_marginal_obs = bnet.get_marginal_by_brute_force([a, b])
    bc_marginal_obs = bnet.get_marginal_by_brute_force([b, c])
    ac_marginal_obs = bnet.get_marginal_by_brute_force([a, c])

    assert fuzzy_match(a_marginal_true, a_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(b_marginal_true, b_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(c_marginal_true, c_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ab_marginal_true, ab_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(bc_marginal_true, bc_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ac_marginal_true, ac_marginal_obs, FUZZY_MATCH_ERROR)

    print('predicted marginal is {}'.format(c_marginal_obs))
    s = 0
    for _ in range(N_SAMPLE):
        s += bnet.sample()[c]
    print('estimated marginal is {}'.format(s * 1.0 / N_SAMPLE))
    print()


def test_marginals_brute_force_v_structure(rng):
    print('--- TESTING V_STRUCTURE ---')
    pa1 = rng.uniform()
    pb1 = rng.uniform()
    pa0b0_c1 = rng.uniform()
    pa1b0_c1 = rng.uniform()
    pa0b1_c1 = rng.uniform()
    pa1b1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    b = BinaryNode('b', [], [pb1], rng=rng)
    c = BinaryNode(
        'c',
        [a, b],
        [pa0b0_c1, pa1b0_c1, pa0b1_c1, pa1b1_c1],
        rng=rng
    )
    bnet = BinaryBayesianNetwork([a, b, c])

    a_marginal_true = pa1
    b_marginal_true = pb1
    c_marginal_true = (
        (1 - pa1) * (1 - pb1) * pa0b0_c1
        + pa1 * (1 - pb1) * pa1b0_c1
        + (1 - pa1) * pb1 * pa0b1_c1
        + pa1 * pb1 * pa1b1_c1
    )
    ab_marginal_true = [
        (1 - pa1) * (1 - pb1),
        pa1 * (1 - pb1),
        (1 - pa1) * pb1,
        pa1 * pb1
    ]
    bc_marginal_true = [
        (1 - pa1) * (1 - pb1) * (1 - pa0b0_c1)
            + pa1 * (1 - pb1) * (1 - pa1b0_c1),
        (1 - pa1) * pb1 * (1 - pa0b1_c1) + pa1 * pb1 * (1 - pa1b1_c1),
        (1 - pa1) * (1 - pb1) * pa0b0_c1 + pa1 * (1 - pb1) * pa1b0_c1,
        (1 - pa1) * pb1 * pa0b1_c1 + pa1 * pb1 * pa1b1_c1
    ]
    ac_marginal_true = [
        (1 - pb1) * (1 - pa1) * (1 - pa0b0_c1)
            + pb1 * (1 - pa1) * (1 - pa0b1_c1),
        (1 - pb1) * pa1 * (1 - pa1b0_c1) + pb1 * pa1 * (1 - pa1b1_c1),
        (1 - pb1) * (1 - pa1) * pa0b0_c1 + pb1 * (1 - pa1) * pa0b1_c1,
        (1 - pb1) * pa1 * pa1b0_c1 + pb1 * pa1 * pa1b1_c1
    ]
    a_marginal_obs = bnet.get_marginal_by_brute_force([a])[1]
    b_marginal_obs = bnet.get_marginal_by_brute_force([b])[1]
    c_marginal_obs = bnet.get_marginal_by_brute_force([c])[1]
    ab_marginal_obs = bnet.get_marginal_by_brute_force([a, b])
    bc_marginal_obs = bnet.get_marginal_by_brute_force([b, c])
    ac_marginal_obs = bnet.get_marginal_by_brute_force([a, c])

    assert fuzzy_match(a_marginal_true, a_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(b_marginal_true, b_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(c_marginal_true, c_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ab_marginal_true, ab_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(bc_marginal_true, bc_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ac_marginal_true, ac_marginal_obs, FUZZY_MATCH_ERROR)

    print('predicted marginal is {}'.format(c_marginal_obs))
    s = 0
    for _ in range(N_SAMPLE):
        s += bnet.sample()[c]
    print('estimated marginal is {}'.format(s * 1.0 / N_SAMPLE))
    print()


def test_marginals_brute_force_triangle(rng):
    print('--- TESTING TRIANGLE ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pa0b0_c1 = rng.uniform()
    pa1b0_c1 = rng.uniform()
    pa0b1_c1 = rng.uniform()
    pa1b1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1], rng=rng)
    c = BinaryNode(
        'c',
        [a, b],
        [pa0b0_c1, pa1b0_c1, pa0b1_c1, pa1b1_c1],
        rng=rng
    )
    bnet = BinaryBayesianNetwork([a, b, c])

    a_marginal_true = pa1
    b_marginal_true = (1 - pa1) * pa0_b1 + pa1 * pa1_b1
    c_marginal_true = (
        (1 - pa1) * (1 - pa0_b1) * pa0b0_c1
        + pa1 * (1 - pa1_b1) * pa1b0_c1
        + (1 - pa1) * pa0_b1 * pa0b1_c1
        + pa1 * pa1_b1 * pa1b1_c1
    )
    ab_marginal_true = [
        (1 - pa1) * (1 - pa0_b1),
        pa1 * (1 - pa1_b1),
        (1 - pa1) * pa0_b1,
        pa1 * pa1_b1
    ]
    bc_marginal_true = [
        (1 - pa1) * (1 - pa0_b1) * (1 - pa0b0_c1)
            + pa1 * (1 - pa1_b1) * (1 - pa1b0_c1),
        (1 - pa1) * pa0_b1 * (1 - pa0b1_c1) + pa1 * pa1_b1 * (1 - pa1b1_c1),
        (1 - pa1) * (1 - pa0_b1) * pa0b0_c1 + pa1 * (1 - pa1_b1) * pa1b0_c1,
        (1 - pa1) * pa0_b1 * pa0b1_c1 + pa1 * pa1_b1 * pa1b1_c1
    ]
    ac_marginal_true = [
        (1 - pa1) * (1 - pa0_b1) * (1 - pa0b0_c1)
            + (1 - pa1) * pa0_b1 * (1 - pa0b1_c1),
        pa1 * (1 - pa1_b1) * (1 - pa1b0_c1) + pa1 * pa1_b1 * (1 - pa1b1_c1),
        (1 - pa1) * (1 - pa0_b1) * pa0b0_c1 + (1 - pa1) * pa0_b1 * pa0b1_c1,
        pa1 * (1 - pa1_b1) * pa1b0_c1 + pa1 * pa1_b1 * pa1b1_c1
    ]
    a_marginal_obs = bnet.get_marginal_by_brute_force([a])[1]
    b_marginal_obs = bnet.get_marginal_by_brute_force([b])[1]
    c_marginal_obs = bnet.get_marginal_by_brute_force([c])[1]
    ab_marginal_obs = bnet.get_marginal_by_brute_force([a, b])
    bc_marginal_obs = bnet.get_marginal_by_brute_force([b, c])
    ac_marginal_obs = bnet.get_marginal_by_brute_force([a, c])

    assert fuzzy_match(a_marginal_true, a_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(b_marginal_true, b_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(c_marginal_true, c_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ab_marginal_true, ab_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(bc_marginal_true, bc_marginal_obs, FUZZY_MATCH_ERROR)
    assert fuzzy_match(ac_marginal_true, ac_marginal_obs, FUZZY_MATCH_ERROR)

    print('predicted marginal is {}'.format(c_marginal_obs))
    s = 0
    for _ in range(N_SAMPLE):
        s += bnet.sample()[c]
    print('estimated marginal is {}'.format(s * 1.0 / N_SAMPLE))
    print()


def test_marginals(rng):
    print('--- TESTING NODE MARGINALS ---')
    test_marginals_brute_force_cascade(rng)
    test_marginals_brute_force_common_parent(rng)
    test_marginals_brute_force_v_structure(rng)
    test_marginals_brute_force_triangle(rng)


if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    test_marginals(rng)

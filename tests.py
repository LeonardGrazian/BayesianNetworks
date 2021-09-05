
import numpy as np
seed = np.random.randint(1, 100)
rng = np.random.default_rng(seed)
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork

# constants
N_SAMPLE = 10000
CORRELATED_THRESHOLD = 0.1
NOT_CORRELATED_THRESHOLD = 0.05
FUZZY_MATCH_ERROR = 0.0001


# @param true_val: float
# @param obs_val: float
# @param error: float
# returns True if obs_val is within error of true_val
def fuzzy_match(true_val, obs_val, error):
    assert type(true_val) == type(obs_val)
    if isinstance(true_val, list):
        for tv, ov in zip(true_val, obs_val):
            if ov < tv * (1.0 - error) or ov > tv * (1.0 + error):
                return False
        return True
    else:
        return (
            obs_val >= true_val * (1.0 - error)
            and obs_val <= true_val * (1.0 + error)
        )


def test_independence_cascade():
    print('--- TESTING CASCADE ---')
    a = BinaryNode('a', [], [0.75])
    b = BinaryNode('b', [a], [0.9, 0.1])
    c = BinaryNode('c', [b], [0.7, 0.9])
    bnet = BinaryBayesianNetwork([a, b, c])

    # test that if b is unobserved then a and c are dependent
    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.sample()
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) > CORRELATED_THRESHOLD
    print('(b unobserved), a-c correlation is {:.2} (dependent)'.format(corr))

    # test that if b is observed then a and c are independent
    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({b: 0})
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) < NOT_CORRELATED_THRESHOLD
    print('(b=0), a-c correlation is {:.2} (independent)'.format(corr))

    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({b: 1})
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) < NOT_CORRELATED_THRESHOLD
    print('(b=1), a-c correlation is {:.2} (independent)'.format(corr))
    print()


def test_independence_common_parent():
    print('--- TESTING COMMON_PARENT ---')
    b = BinaryNode('b', [], [0.75])
    a = BinaryNode('a', [b], [0.9, 0.1])
    c = BinaryNode('c', [b], [0.7, 0.9])
    bnet = BinaryBayesianNetwork([a, b, c])

    # test that when b is unobserved, a and c are dependent
    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.sample()
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) > CORRELATED_THRESHOLD
    print('(b unobserved), a-c correlation is {:.2} (dependent)'.format(corr))

    # test that when b is observed, a and c are independent
    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({b: 0})
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) < NOT_CORRELATED_THRESHOLD
    print('(b=0), a-c correlation is {:.2} (independent)'.format(corr))

    a_vals = []
    c_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({b: 1})
        a_vals.append(s[a])
        c_vals.append(s[c])
    corr = np.corrcoef(a_vals, c_vals)[0, 1]
    assert abs(corr) < NOT_CORRELATED_THRESHOLD
    print('(b=1), a-c correlation is {:.2} (independent)'.format(corr))
    print()


def test_independence_v_structure():
    print('--- TESTING V_STRUCTURE ---')
    a = BinaryNode('a', [], [0.75])
    b = BinaryNode('b', [], [0.9])
    c = BinaryNode('c', [a, b], [0.7, 0.9, 0.6, 0.3])
    bnet = BinaryBayesianNetwork([a, b, c])

    # test that when c is unobserved, a and b are independent
    a_vals = []
    b_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.sample()
        a_vals.append(s[a])
        b_vals.append(s[b])
    corr = np.corrcoef(a_vals, b_vals)[0, 1]
    assert abs(corr) < NOT_CORRELATED_THRESHOLD
    print('(c unobserved), a-b correlation is {:.2} (independent)'.format(corr))

    # test that when c is observed, a and b are dependent
    a_vals = []
    b_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({c: 0})
        a_vals.append(s[a])
        b_vals.append(s[b])
    corr = np.corrcoef(a_vals, b_vals)[0, 1]
    assert abs(corr) > CORRELATED_THRESHOLD
    print('(c=0), a-b correlation is {:.2} (dependent)'.format(corr))

    a_vals = []
    b_vals = []
    for _ in range(N_SAMPLE):
        s = bnet.conditional_sample({c: 1})
        a_vals.append(s[a])
        b_vals.append(s[b])
    corr = np.corrcoef(a_vals, b_vals)[0, 1]
    assert abs(corr) > CORRELATED_THRESHOLD
    print('(c=1), a-b correlation is {:.2} (dependent)'.format(corr))
    print()


def test_independence():
    print('--- TESTING NODE INDEPENDENCE ---')
    test_independence_cascade()
    test_independence_common_parent()
    test_independence_v_structure()


def test_marginals_brute_force_cascade():
    print('--- TESTING CASCADE ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pb0_c1 = rng.uniform()
    pb1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1])
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1])
    c = BinaryNode('c', [b], [pb0_c1, pb1_c1])
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


def test_marginals_brute_force_common_parent():
    print('--- TESTING COMMON_PARENT ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pa0_c1 = rng.uniform()
    pa1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1])
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1])
    c = BinaryNode('c', [a], [pa0_c1, pa1_c1])
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


def test_marginals_brute_force_v_structure():
    print('--- TESTING V_STRUCTURE ---')
    pa1 = rng.uniform()
    pb1 = rng.uniform()
    pa0b0_c1 = rng.uniform()
    pa1b0_c1 = rng.uniform()
    pa0b1_c1 = rng.uniform()
    pa1b1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1])
    b = BinaryNode('b', [], [pb1])
    c = BinaryNode('c', [a, b], [pa0b0_c1, pa1b0_c1, pa0b1_c1, pa1b1_c1])
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


def test_marginals_brute_force_triangle():
    print('--- TESTING TRIANGLE ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pa0b0_c1 = rng.uniform()
    pa1b0_c1 = rng.uniform()
    pa0b1_c1 = rng.uniform()
    pa1b1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1])
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1])
    c = BinaryNode('c', [a, b], [pa0b0_c1, pa1b0_c1, pa0b1_c1, pa1b1_c1])
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


def test_marginals():
    print('--- TESTING NODE MARGINALS ---')
    test_marginals_brute_force_cascade()
    test_marginals_brute_force_common_parent()
    test_marginals_brute_force_v_structure()
    test_marginals_brute_force_triangle()


def run_tests():
    print('--- RUNNING TESTS WITH SEED {} ---'.format(seed))
    test_independence()
    test_marginals()


if __name__ == '__main__':
    run_tests()
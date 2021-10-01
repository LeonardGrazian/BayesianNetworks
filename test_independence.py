
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork

# constants
N_SAMPLE = 10000
CORRELATED_THRESHOLD = 0.1
NOT_CORRELATED_THRESHOLD = 0.05


def test_independence_cascade(rng):
    print('--- TESTING CASCADE ---')
    a = BinaryNode('a', [], [0.75], rng=rng)
    b = BinaryNode('b', [a], [0.9, 0.1], rng=rng)
    c = BinaryNode('c', [b], [0.7, 0.9], rng=rng)
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


def test_independence_common_parent(rng):
    print('--- TESTING COMMON_PARENT ---')
    b = BinaryNode('b', [], [0.75], rng=rng)
    a = BinaryNode('a', [b], [0.9, 0.1], rng=rng)
    c = BinaryNode('c', [b], [0.7, 0.9], rng=rng)
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


def test_independence_v_structure(rng):
    print('--- TESTING V_STRUCTURE ---')
    a = BinaryNode('a', [], [0.75], rng=rng)
    b = BinaryNode('b', [], [0.9], rng=rng)
    c = BinaryNode('c', [a, b], [0.7, 0.9, 0.6, 0.3], rng=rng)
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


def test_independence(rng):
    print('--- TESTING NODE INDEPENDENCE ---')
    test_independence_cascade(rng)
    test_independence_common_parent(rng)
    test_independence_v_structure(rng)


if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    print('--- RUNNING TESTS WITH SEED {} ---'.format(seed))
    test_independence(rng)

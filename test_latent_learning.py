
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork
from utils import fuzzy_match, fuzzy_match_add

# constants
LEARNING_DATA_SIZE = 1000 # int(1.0e6)
LEARNING_ERROR = 0.05
LEARNING_ERROR_ADD = 0.05


def test_latent_learning_cascade(rng):
    print('--- TESTING CASCADE ---')
    # create a "true" bayesian network to generate our data
    pa1_true = rng.uniform()
    pa0_b1_true = rng.uniform()
    pa1_b1_true = rng.uniform()
    pb0_c1_true = rng.uniform()
    pb1_c1_true = rng.uniform()
    a_true = BinaryNode('a', [], [pa1_true], rng=rng)
    b_true = BinaryNode('b', [a_true], [pa0_b1_true, pa1_b1_true], rng=rng)
    c_true = BinaryNode('c', [b_true], [pb0_c1_true, pb1_c1_true], rng=rng)
    bnet_true = BinaryBayesianNetwork([a_true, b_true, c_true])

    # generate data
    data = []
    a_vals_true = []
    c_vals_true = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        del obs[b_true]
        data.append(obs)
        a_vals_true.append(obs[a_true])
        c_vals_true.append(obs[c_true])

    # initialize new bayesian net for learning
    pa1_init = rng.uniform()
    pa0_b1_init = rng.uniform()
    pa1_b1_init = rng.uniform()
    pb0_c1_init = rng.uniform()
    pb1_c1_init = rng.uniform()
    a = BinaryNode('a', [], [pa1_init], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1_init, pa1_b1_init], rng=rng)
    c = BinaryNode('c', [b], [pb0_c1_init, pb1_c1_init], rng=rng)
    bnet = BinaryBayesianNetwork([a, b, c])

    # translate true nodes to learning nodes in generated data
    node_mapping = {a_true: a, b_true: b, c_true: c}
    data = [{node_mapping[k]: v for k, v in obs.items()} for obs in data]

    # learning
    bnet.learn_latent([b], data)

    # check learning results
    a_vals = []
    c_vals = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet.sample()
        a_vals.append(obs[a])
        c_vals.append(obs[c])

    pa1 = sum(a_vals) * 1.0 / LEARNING_DATA_SIZE
    pa1_true = sum(a_vals_true) * 1.0 / LEARNING_DATA_SIZE
    print('learned p(b) = {:.3}'.format(pa1))
    print('true p(b) = {:.3}'.format(pa1_true))
    print()
    pc1 = sum(c_vals) * 1.0 / LEARNING_DATA_SIZE
    pc1_true = sum(c_vals_true) * 1.0 / LEARNING_DATA_SIZE
    print('learned p(c) = {:.3}'.format(pc1))
    print('true p(c) = {:.3}'.format(pc1_true))
    print()
    ac_corr = np.corrcoef(a_vals, c_vals)[0, 1]
    ac_corr_true = np.corrcoef(a_vals_true, c_vals_true)[0, 1]
    print('learned a-c corr = {:.3}'.format(ac_corr))
    print('true a-c corr = {:.3}'.format(ac_corr_true))
    print()

    assert fuzzy_match(pa1_true, pa1, LEARNING_ERROR)
    assert fuzzy_match(pc1_true, pc1_true, LEARNING_ERROR)
    assert fuzzy_match_add(ac_corr_true, ac_corr, LEARNING_ERROR_ADD)


def test_latent_learning_common_parent(rng):
    print('--- TESTING COMMON_PARENT ---')
    # create a "true" bayesian network to generate our data
    pa1_true = rng.uniform()
    pa0_b1_true = rng.uniform()
    pa1_b1_true = rng.uniform()
    pa0_c1_true = rng.uniform()
    pa1_c1_true = rng.uniform()
    a_true = BinaryNode('a', [], [pa1_true], rng=rng)
    b_true = BinaryNode('b', [a_true], [pa0_b1_true, pa1_b1_true], rng=rng)
    c_true = BinaryNode('c', [a_true], [pa0_c1_true, pa1_c1_true], rng=rng)
    bnet_true = BinaryBayesianNetwork([a_true, b_true, c_true])

    # generate data
    data = []
    b_vals_true = []
    c_vals_true = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        del obs[a_true]
        data.append(obs)
        b_vals_true.append(obs[b_true])
        c_vals_true.append(obs[c_true])

    # initialize new bayesian net for learning
    pa1_init = rng.uniform()
    pa0_b1_init = rng.uniform()
    pa1_b1_init = rng.uniform()
    pa0_c1_init = rng.uniform()
    pa1_c1_init = rng.uniform()
    a = BinaryNode('a', [], [pa1_init], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1_init, pa1_b1_init], rng=rng)
    c = BinaryNode('c', [a], [pa0_c1_init, pa1_c1_init], rng=rng)
    bnet = BinaryBayesianNetwork([a, b, c])

    # translate true nodes to learning nodes in generated data
    node_mapping = {a_true: a, b_true: b, c_true: c}
    data = [{node_mapping[k]: v for k, v in obs.items()} for obs in data]

    # learning
    bnet.learn_latent([a], data)

    # check learning results
    b_vals = []
    c_vals = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet.sample()
        b_vals.append(obs[b])
        c_vals.append(obs[c])

    pb1 = sum(b_vals) * 1.0 / LEARNING_DATA_SIZE
    pb1_true = sum(b_vals_true) * 1.0 / LEARNING_DATA_SIZE
    print('learned p(b) = {:.3}'.format(pb1))
    print('true p(b) = {:.3}'.format(pb1_true))
    print()
    pc1 = sum(c_vals) * 1.0 / LEARNING_DATA_SIZE
    pc1_true = sum(c_vals_true) * 1.0 / LEARNING_DATA_SIZE
    print('learned p(c) = {:.3}'.format(pc1))
    print('true p(c) = {:.3}'.format(pc1_true))
    print()
    bc_corr = np.corrcoef(b_vals, c_vals)[0, 1]
    bc_corr_true = np.corrcoef(b_vals_true, c_vals_true)[0, 1]
    print('learned b-c corr = {:.3}'.format(bc_corr))
    print('true b-c corr = {:.3}'.format(bc_corr_true))
    print()

    assert fuzzy_match(pb1_true, pb1, LEARNING_ERROR)
    assert fuzzy_match(pc1_true, pc1_true, LEARNING_ERROR)
    assert fuzzy_match_add(bc_corr_true, bc_corr, LEARNING_ERROR_ADD)


def test_latent_learning(rng):
    print('--- TESTING LATENT LEARNING ---')
    test_latent_learning_cascade(rng)
    test_latent_learning_common_parent(rng)


if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    print('--- RUNNING TESTS WITH SEED {} ---'.format(seed))
    test_latent_learning(rng)

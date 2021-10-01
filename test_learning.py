
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork
from utils import fuzzy_match

# constants
FUZZY_MATCH_ERROR = 0.0001
LEARNING_DATA_SIZE = int(1.0e6)
LEARNING_ERROR = 0.05


def test_learning_node_level(rng):
    print('--- TESTING NODE_LEVEL ---')
    pa1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    assert a.probability(()) == pa1

    new_pa1 = round(rng.uniform(), 1)
    data = {(): (10, int(10 * new_pa1))}
    a.learn(data)
    assert a.probability(()) == new_pa1
    print('PASSED')
    print()


def test_learning_single_node(rng):
    print('--- TESTING SINGLE_NODE ---')
    pa1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    bnet = BinaryBayesianNetwork([a])
    assert bnet.probability({a: 1}) == pa1

    n_a1s = rng.integers(1, 10)
    data = (
        [{a: 0} for _ in range(10 - n_a1s)]
        + [{a: 1} for _ in range(n_a1s)]
    )
    bnet.learn(data)
    assert fuzzy_match(
        n_a1s * 1.0 / 10.0,
        bnet.probability({a: 1}),
        FUZZY_MATCH_ERROR
    )
    print('PASSED')
    print()


def test_learning_two_nodes(rng):
    print('--- TESTING TWO_NODES ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    a = BinaryNode('a', [], [pa1], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1], rng=rng)
    bnet = BinaryBayesianNetwork([a, b])
    assert bnet.probability({a: 1}) == pa1
    assert fuzzy_match(
        (1 - pa1) * pa0_b1,
        bnet.probability({a: 0, b: 1}),
        FUZZY_MATCH_ERROR
    )
    assert fuzzy_match(
        pa1 * pa1_b1,
        bnet.probability({a: 1, b: 1}),
        FUZZY_MATCH_ERROR
    )
    assert fuzzy_match(
        (1 - pa1) * pa0_b1 + pa1 * pa1_b1,
        bnet.get_marginal_by_brute_force([b])[1],
        FUZZY_MATCH_ERROR
    )

    n_a1s = rng.integers(1, 10)
    n_a0_b1s = rng.integers(1, 10)
    n_a1_b1s = rng.integers(1, 10)
    new_pa1 = n_a1s * 1.0 / 10.0
    new_pa0_b1 = n_a0_b1s * 1.0 / 10.0
    new_pa1_b1 = n_a1_b1s * 1.0 / 10.0
    nested_data = (
        [
            [
                {a: 0, b: 0}
                for _ in range(10 - n_a0_b1s)
            ]
            + [
                {a: 0, b: 1}
                for _ in range(n_a0_b1s)
            ]
            for _ in range(10 - n_a1s)
        ]
        + [
            [
                {a: 1, b: 0}
                for _ in range(10 - n_a1_b1s)
            ]
            + [
                {a: 1, b: 1}
                for _ in range(n_a1_b1s)
            ]
            for _ in range(n_a1s)
        ]
    )
    flat_data = []
    for data_list in nested_data:
        for d in data_list:
            flat_data.append(d)
    bnet.learn(flat_data)
    assert fuzzy_match(
        (1 - new_pa1) * new_pa0_b1,
        bnet.probability({a: 0, b: 1}),
        FUZZY_MATCH_ERROR
    )
    assert fuzzy_match(
        new_pa1 * new_pa1_b1,
        bnet.probability({a: 1, b: 1}),
        FUZZY_MATCH_ERROR
    )
    assert fuzzy_match(
        (1 - new_pa1) * new_pa0_b1 + new_pa1 * new_pa1_b1,
        bnet.get_marginal_by_brute_force([b])[1],
        FUZZY_MATCH_ERROR
    )
    print('PASSED')
    print()


def test_learning_cascade(rng):
    print('--- TESTING LEARNING_CASCADE ---')
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
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        data.append(obs)

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

    # learn from generated data to replicate "true" bayesian net
    bnet.learn(data)

    # check learning results
    for p_true, p in zip(a_true.prob_table, a.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(b_true.prob_table, b.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(c_true.prob_table, c.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    print('PASSED')
    print()


def test_learning_common_parent(rng):
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
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        data.append(obs)

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

    # learn from generated data to replicate "true" bayesian net
    bnet.learn(data)

    # check learning results
    for p_true, p in zip(a_true.prob_table, a.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(b_true.prob_table, b.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(c_true.prob_table, c.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    print('PASSED')
    print()


def test_learning_v_structure(rng):
    print('--- TESTING V_STRUCTURE ---')
    # create a "true" bayesian network to generate our data
    pa1_true = rng.uniform()
    pb1_true = rng.uniform()
    pa0b0_c1_true = rng.uniform()
    pa1b0_c1_true = rng.uniform()
    pa0b1_c1_true = rng.uniform()
    pa1b1_c1_true = rng.uniform()
    a_true = BinaryNode('a', [], [pa1_true], rng=rng)
    b_true = BinaryNode('b', [], [pb1_true], rng=rng)
    c_true = BinaryNode(
        'c',
        [a_true, b_true],
        [pa0b0_c1_true, pa1b0_c1_true, pa0b1_c1_true, pa1b1_c1_true],
        rng=rng
    )
    bnet_true = BinaryBayesianNetwork([a_true, b_true, c_true])

    # generate data
    data = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        data.append(obs)

    # initialize new bayesian net for learning
    pa1_init = rng.uniform()
    pb1_init = rng.uniform()
    pa0b0_c1_init = rng.uniform()
    pa1b0_c1_init = rng.uniform()
    pa0b1_c1_init = rng.uniform()
    pa1b1_c1_init = rng.uniform()
    a = BinaryNode('a', [], [pa1_init], rng=rng)
    b = BinaryNode('b', [], [pb1_init], rng=rng)
    c = BinaryNode(
        'c',
        [a, b],
        [pa0b0_c1_init, pa1b0_c1_init, pa0b1_c1_init, pa1b1_c1_init],
        rng=rng
    )
    bnet = BinaryBayesianNetwork([a, b, c])

    # translate true nodes to learning nodes in generated data
    node_mapping = {a_true: a, b_true: b, c_true: c}
    data = [{node_mapping[k]: v for k, v in obs.items()} for obs in data]

    # learn from generated data to replicate "true" bayesian net
    bnet.learn(data)

    # check learning results
    for p_true, p in zip(a_true.prob_table, a.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(b_true.prob_table, b.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(c_true.prob_table, c.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    print('PASSED')
    print()


def test_learning_triangle(rng):
    print('--- TESTING TRIANGLE ---')
    # create a "true" bayesian network to generate our data
    pa1_true = rng.uniform()
    pa0_b1_true = rng.uniform()
    pa1_b1_true = rng.uniform()
    pa0b0_c1_true = rng.uniform()
    pa1b0_c1_true = rng.uniform()
    pa0b1_c1_true = rng.uniform()
    pa1b1_c1_true = rng.uniform()
    a_true = BinaryNode('a', [], [pa1_true], rng=rng)
    b_true = BinaryNode('b', [a_true], [pa0_b1_true, pa1_b1_true], rng=rng)
    c_true = BinaryNode(
        'c',
        [a_true, b_true],
        [pa0b0_c1_true, pa1b0_c1_true, pa0b1_c1_true, pa1b1_c1_true],
        rng=rng
    )
    bnet_true = BinaryBayesianNetwork([a_true, b_true, c_true])

    # generate data
    data = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        data.append(obs)

    # initialize new bayesian net for learning
    pa1_init = rng.uniform()
    pa0_b1_init = rng.uniform()
    pa1_b1_init = rng.uniform()
    pa0b0_c1_init = rng.uniform()
    pa1b0_c1_init = rng.uniform()
    pa0b1_c1_init = rng.uniform()
    pa1b1_c1_init = rng.uniform()
    a = BinaryNode('a', [], [pa1_init], rng=rng)
    b = BinaryNode('b', [a], [pa0_b1_init, pa1_b1_init], rng=rng)
    c = BinaryNode(
        'c',
        [a, b],
        [pa0b0_c1_init, pa1b0_c1_init, pa0b1_c1_init, pa1b1_c1_init],
        rng=rng
    )
    bnet = BinaryBayesianNetwork([a, b, c])

    # translate true nodes to learning nodes in generated data
    node_mapping = {a_true: a, b_true: b, c_true: c}
    data = [{node_mapping[k]: v for k, v in obs.items()} for obs in data]

    # learn from generated data to replicate "true" bayesian net
    bnet.learn(data)

    # check learning results
    for p_true, p in zip(a_true.prob_table, a.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(b_true.prob_table, b.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    for p_true, p in zip(c_true.prob_table, c.prob_table):
        assert fuzzy_match(p_true, p, LEARNING_ERROR)
    print('PASSED')
    print()


def test_learning(rng):
    print('--- TESTING LEARNING ---')
    test_learning_node_level(rng)
    test_learning_single_node(rng)
    test_learning_two_nodes(rng)
    test_learning_cascade(rng)
    test_learning_common_parent(rng)
    test_learning_v_structure(rng)
    test_learning_triangle(rng)


if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    print('--- RUNNING TESTS WITH SEED {} ---'.format(seed))
    test_learning(rng)

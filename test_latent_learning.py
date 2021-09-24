
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork

# constants
LEARNING_DATA_SIZE = int(1.0e6)
LEARNING_ERROR = 0.05


def test_latent_learning(rng):
    print('--- TESTING LATENT LEARNING ---')
    # create a "true" bayesian network to generate our data
    pa1_true = rng.uniform()
    pa0_b1_true = rng.uniform()
    pa1_b1_true = rng.uniform()
    pb0_c1_true = rng.uniform()
    pb1_c1_true = rng.uniform()
    a_true = BinaryNode('a', [], [pa1_true])
    b_true = BinaryNode('b', [a_true], [pa0_b1_true, pa1_b1_true])
    c_true = BinaryNode('c', [b_true], [pb0_c1_true, pb1_c1_true])
    bnet_true = BinaryBayesianNetwork([a_true, b_true, c_true])

    # generate data
    data = []
    for _ in range(LEARNING_DATA_SIZE):
        obs = bnet_true.sample()
        del obs[b_true]
        data.append(obs)

    # initialize new bayesian net for learning
    pa1_init = rng.uniform()
    pa0_b1_init = rng.uniform()
    pa1_b1_init = rng.uniform()
    pb0_c1_init = rng.uniform()
    pb1_c1_init = rng.uniform()
    a = BinaryNode('a', [], [pa1_init])
    b = BinaryNode('b', [a], [pa0_b1_init, pa1_b1_init])
    c = BinaryNode('c', [b], [pb0_c1_init, pb1_c1_init])
    bnet = BinaryBayesianNetwork([a, b, c])

    # translate true nodes to learning nodes in generated data
    node_mapping = {a_true: a, b_true: b, c_true: c}
    data = [{node_mapping[k]: v for k, v in obs.items()} for obs in data]

    # learn from generated data to replicate "true" bayesian net
    bnet.learn_latent(data)

    # check learning results
    print(a.prob_table)
    print(a_true.prob_table)
    print()
    print(b.prob_table)
    print(b_true.prob_table)
    print()
    print(c.prob_table)
    print(c_true.prob_table)
    print()


if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    test_latent_learning(rng)

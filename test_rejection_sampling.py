
import numpy as np
from binary_bayesian_network import BinaryNode, BinaryBayesianNetwork
from utils import fuzzy_match

# constants
REJECTION_SAMPLING_SAMPLES = int(1.0e5)
REJECTION_SAMPLING_ERROR = 0.01


def test_rejection_sampling(rng):
    print('--- TESTING REJECTION SAMPLING ---')
    pa1 = rng.uniform()
    pa0_b1 = rng.uniform()
    pa1_b1 = rng.uniform()
    pb0_c1 = rng.uniform()
    pb1_c1 = rng.uniform()
    a = BinaryNode('a', [], [pa1])
    b = BinaryNode('b', [a], [pa0_b1, pa1_b1])
    c = BinaryNode('c', [b], [pb0_c1, pb1_c1])
    bnet = BinaryBayesianNetwork([a, b, c])

    positives = 0
    for _ in range(REJECTION_SAMPLING_SAMPLES):
        node_samples = bnet.sample(node_values={a: 0})
        if node_samples is not None:
            positives += 1
    sample_prob = positives * 1.0 / REJECTION_SAMPLING_SAMPLES
    true_prob = bnet.probability({a: 0})
    print('rejection sampling prob = {:.2} (should be {:.2})'.format(
        sample_prob,
        true_prob
    ))
    assert fuzzy_match(true_prob, sample_prob, REJECTION_SAMPLING_ERROR)

if __name__ == '__main__':
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    test_rejection_sampling(rng)

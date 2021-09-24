
import numpy as np

from utils import test_utils
from test_independence import test_independence
from test_marginals import test_marginals
from test_learning import test_learning
from test_latent_learning import test_latent_learning
from test_rejection_sampling import test_rejection_sampling


def run_tests():
    seed = np.random.randint(1, 100)
    rng = np.random.default_rng(seed)
    print('--- RUNNING TESTS WITH SEED {} ---'.format(seed))

    test_utils()
    test_independence()
    test_marginals(rng)
    test_learning(rng)
    # test_latent_learning(rng)
    test_rejection_sampling(rng)


if __name__ == '__main__':
    run_tests()

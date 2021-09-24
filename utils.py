
# @param true_val: float
# @param obs_val: float
# @param error: float
# returns True if obs_val is within error of true_val
def fuzzy_match(true_val, obs_val, error):
    if isinstance(true_val, list):
        assert isinstance(obs_val, list)
        for tv, ov in zip(true_val, obs_val):
            if ov < tv * (1.0 - error) or ov > tv * (1.0 + error):
                return False
        return True
    else:
        return (
            obs_val >= true_val * (1.0 - error)
            and obs_val <= true_val * (1.0 + error)
        )


def test_utils():
    print('--- TESTING UTILS -- ')
    assert fuzzy_match(1.00, 1.05, 0.1)
    assert not fuzzy_match(1.00, 1.10, 0.05)
    print('PASSED')
    print()


if __name__ == '__main__':
    test_utils()

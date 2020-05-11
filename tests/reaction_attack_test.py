import unittest
import mcelice_qcldpc.reaction_attack as ra


class reaction_attack_tests(unittest.TestCase):
    n0 = 2
    r = 50

    def test_attack(self):
        initialization = ra.reaction_attack(self.n0, self.r)

        h1 = initialization.key_recovery(10)

        assert len(h1) == self.r

        print("test_random_vector_with_fixed_wight")

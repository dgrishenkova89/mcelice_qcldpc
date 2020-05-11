import unittest
import mcelice_qcldpc.McElieceSystem as MC
import mcelice_qcldpc.reaction_attack as ra


class reaction_attack_tests(unittest.TestCase):
    n0 = 2
    r = 80
    w = 12
    m = 6
    t = 15

    def test_attack(self):
        cryptosystem = MC.McElieceSystem(self.n0, self.r, self.w, self.m, self.t)
        source_cipher = cryptosystem.encode()
        source_message = cryptosystem.message
        source_private_key = cryptosystem.LDPC.H_matrix[0]

        h = []
        for i in range(1000):
            initialization = ra.reaction_attack(self.n0, self.r, cryptosystem)
            success, h1 = initialization.key_recovery(1000)
            if success:
                h.append(h1)

        assert len(h1) != 0

        print("test_random_vector_with_fixed_wight")

import unittest
import mcelice_qcldpc.McElieceSystem as MC
import mcelice_qcldpc.reaction_attack as ra


class reaction_attack_tests(unittest.TestCase):
    n0 = 2
    r = 4801
    w = 9
    m = 2
    t = 95

    def test_attack(self):
        cryptosystem = MC.McElieceSystem(self.n0, self.r, self.w, self.m, self.t)

        h = []
        cipher_text_list = []
        success_h = []
        success_p0 = []
        success_p1 = []
        success_s0 = []
        success_s1 = []
        for i in range(100):
            print(i)
            try:
                initialization = ra.reaction_attack(self.n0, self.r, cryptosystem)
                success, h1, cipher_text, p0_0, p1_1, s0_0, s1_1, decode_success = initialization.key_recovery()
                if decode_success:
                    cipher_text_list.append(cipher_text)
                    success_h.append(h1)
                    success_p0.append(p0_0)
                    success_p1.append(p1_1)
                    success_s0.append(s0_0)
                    success_s1.append(s1_1)
                    break
            except:
                continue

        assert len(h) != 0

        print("test_random_vector_with_fixed_wight")

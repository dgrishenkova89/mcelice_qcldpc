import unittest
import yaml
import qcldpc as QC_LDPC
import McElieceSystem as MC
from fileutils import write_matrix_in_files, write_list_in_file, remove_files


class QcldpcTests(unittest.TestCase):
    n0 = 2
    r = 80
    w = 7
    m = 3
    t = 21

    def test_random_vector_with_fixed_wight(self):
        vector = QC_LDPC.gen_random_vector_with_fixed_weight(self.r, weight=self.w)

        assert len(vector) != 0

        print("test_random_vector_with_fixed_wight IsTrue: ", len(vector) != 0)

    def test_get_cyclic_matrix(self):
        vector = QC_LDPC.gen_random_vector_with_fixed_weight(self.r, weight=self.w)
        cyclic_matrix = QC_LDPC.get_cyclic_matrix(vector)

        assert len(vector) == self.r
        assert len(cyclic_matrix) == self.r

        print("test_get_cyclic_matrix: True")

    def test_identity_matrix(self):
        I = QC_LDPC.gen_identity_matrix(size=self.r)

        assert len(I) == self.r

        print("test_identity_matrix True")

    def test_gen_scrambling_matrix(self):
        scranble_matrix = QC_LDPC.gen_scrambling_matrix(10)

        assert len(scranble_matrix) == 10
        for i in range(10):
            assert QC_LDPC.get_hamming_weight(scranble_matrix[i]) == 1

    def test_gen_cyclic_matrix(self):
        H = QC_LDPC.gen_qc_cyclic_matrix(size=self.r, weight=self.w)

        assert len(H) == self.r

        print("test_gen_cyclic_matrix True")

    def test_decode(self):
        ldpc = QC_LDPC.LDPC(self.n0, self.r, self.w, self.m, self.t)

        cryptosystem = MC.McElieceSystem(ldpc)
        code = cryptosystem.encode()
        message = cryptosystem.decode(code)

        config = yaml.safe_load(open('config.yml'))
        remove_files(config)
        write_matrix_in_files(config, "generator-matrix", ldpc.G_matrix)
        write_matrix_in_files(config, "parity-check-matrix", ldpc.H_matrix)
        write_list_in_file(config, "public-key", ldpc.public_key[0])
        write_matrix_in_files(config, "private-key", cryptosystem.private_key)
        config.clear()

        assert len(ldpc.H_matrix) == self.r
        assert len(ldpc.G_matrix) == self.r
        assert len(ldpc.G_matrix[0]) == (self.r * self.n0)
        assert len(ldpc.error) == (self.r * self.n0)
        assert len(code) == (self.r * self.n0)
        assert len(message) == len(ldpc.message)
        assert cryptosystem.message.all(message)
import unittest
import yaml
import mcelice_qcldpc.qcldpc as QC_LDPC
import mcelice_qcldpc.McElieceSystem as MC
from mcelice_qcldpc.fileutils import write_matrix_in_files, write_list_in_file, remove_files


class qc_ldpc_tests(unittest.TestCase):
    n0 = 2
    r = 80
    w = 12
    m = 6
    t = 15

    def test_random_vector_with_fixed_wight(self):
        vector = QC_LDPC.gen_random_vector_with_fixed_weight(self.r, weight=self.w)

        assert len(vector) != 0

        print("test_random_vector_with_fixed_wight IsTrue: ", len(vector) != 0)

    def test_identity_matrix(self):
        I = QC_LDPC.gen_identity_matrix(size=self.r)

        assert len(I) == self.r

        print("test_identity_matrix True")

    def test_gen_scrambling_matrix(self):
        scranble_matrix = QC_LDPC.gen_scrambling_matrix(10)

        assert len(scranble_matrix) == 10
        for i in range(10):
            assert QC_LDPC.get_hamming_weight(scranble_matrix[i]) == 1

    def test_decode(self):
        cryptosystem = MC.McElieceSystem(self.n0, self.r, self.w, self.m, self.t)
        code = cryptosystem.encode()
        message, success = cryptosystem.decode(code)

        assert success

        config = yaml.safe_load(open('config.yml'))
        remove_files(config)
        write_matrix_in_files(config, "generator-matrix", cryptosystem.LDPC.G_matrix)
        write_matrix_in_files(config, "parity-check-matrix", cryptosystem.LDPC.H_matrix)
        write_list_in_file(config, "public-key", cryptosystem.LDPC.public_key[0])
        config.clear()

        assert len(cryptosystem.LDPC.H_matrix) == self.r
        assert len(cryptosystem.LDPC.G_matrix) == self.r
        assert len(cryptosystem.LDPC.G_matrix[0]) == (self.r * self.n0)
        assert len(cryptosystem.LDPC.error) == (self.r * self.n0)
        assert len(code) == (self.r * self.n0)
        assert len(message) == len(cryptosystem.LDPC.message)
        assert cryptosystem.message.all(message)
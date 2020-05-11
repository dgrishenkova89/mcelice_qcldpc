import numpy as np
import random
import mcelice_qcldpc.qcldpc as QC_LDPC

class McElieceSystem:

    def __init__(self, n0, r, w, m, t):
        self.LDPC = QC_LDPC.LDPC(n0, r, w, m, t)

    def encode(self):
        print("Start encoding")
        self.message = self.gen_random_message()

        self.error_vector = QC_LDPC.gen_random_vector_with_fixed_weight(self.LDPC.n, self.LDPC.t)
        code = np.array((np.matmul(self.message, self.LDPC.public_key) % 2) ^ self.error_vector)
        print("Finish encoding")
        return code

    def decode(self, code):
        print("Start decoding")
        message, success = self.LDPC.decode_by_gallager(code)

        if not success:
            return message, success

        source_message = np.matmul(message[self.LDPC.r:], self.LDPC.S_matrix)

        print("Finish decoding")
        return source_message, success

    def gen_random_message(self):
        message = np.array([random.choice([0, 1]) for _ in range(self.LDPC.r)])
        return np.array(message)
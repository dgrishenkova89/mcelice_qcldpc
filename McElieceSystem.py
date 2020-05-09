import numpy as np
import random
from mcelice_qcldpc.qcldpc import get_hamming_weight, gen_random_vector_with_fixed_weight

class McElieceSystem:

    def __init__(self, LDPC):
        self.LDPC = LDPC

    def encode(self):
        print("Start encoding")
        self.message = self.gen_random_message()

        error_vector = gen_random_vector_with_fixed_weight(self.LDPC.n, self.LDPC.t // self.LDPC.m)
        code = np.array((np.matmul(self.message, self.LDPC.public_key) + error_vector) % 2)
        print("Finish encoding")
        return code

    def decode(self, code):
        print("Start decoding")
        message = self.LDPC.decode_by_gallager(code)

        weight_source_message = get_hamming_weight(self.message)
        weight_decode_message = get_hamming_weight(message)
        print("Source message weight: {0}".format(weight_source_message))
        print("Decoded message weight: {0}".format(weight_decode_message))
        print("A Weight of source and decoded message is equals: {0}".format(weight_decode_message == weight_source_message))

        print("Finish decoding")
        return message

    def gen_random_message(self):
        message = np.array([random.choice([0, 1])
                            for i in range(self.LDPC.r)])
        return np.array(message)
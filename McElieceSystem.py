import numpy as np


#def rotate_blocks(vector, i):
#    h0 = list(vector[:len(vector) // 2])
#    h1 = list(vector[len(vector) // 2:])
#    h = h0[-i:] + h0[:-i] + h1[-i:] + h1[:-i]
#    return np.array(h)


#def get_syndrom_for_code(code, c):
#    syndrom = []
#    for i in range(len(c) // 2):
#        calculate_syndrom_val = (c + rotate_blocks(c, i)) * code.check_matrix.transpose()
#        syndrom.append(len([i for i in calculate_syndrom_val if i == 1]))
#    return syndrom


class McElieceSystem:

    def __init__(self, LDPC):
        self.LDPC = LDPC

    def encode(self):
        vector = np.array((np.matmul(np.array(self.LDPC.message), self.LDPC.public_key) + self.LDPC.error) % 2)
        print("Encode")
        return vector

    def decode(self, code):
        x_ = np.array(np.matmul(np.array(code), self.LDPC.Q_matrix) % 2)
        u_ = self.LDPC.decode_by_gallager(x_)
        size = len(u_) // 2
        message = (np.matmul(u_[size:], self.LDPC.S_inv) + np.matmul(u_[:size], self.LDPC.S_inv)) % 2
        return message
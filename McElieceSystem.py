import numpy as np


#def rotate_blocks(vector, i):
#    h0 = list(vector[:len(vector) / 2])
#    h1 = list(vector[len(vector) / 2:])
#    h = h0[-i:] + h0[:-i] + h1[-i:] + h1[:-i]
#    return np.array(h)


#def get_syndrom_for_code(code, c):
#    syndrom = []
#    for i in range(len(c) / 2):
#        calculate_syndrom_val = (c + rotate_blocks(c, i)) * code.check_matrix.transpose()
#        syndrom.append(len([i for i in calculate_syndrom_val if i == 1]))
#    return syndrom


class McElieceSystem:

    def __init__(self, LDPC):
        self.LDPC = LDPC
        self.error_vector = LDPC.error
        self.private_key = LDPC.H_matrix
        self.public_key = LDPC.public_key
        self.Q_matrix = LDPC.Q_matrix
        self.S_matrix = LDPC.S_matrix
        self.message = LDPC.message
        self.error = LDPC.error

    def encode(self):
        vector = np.matmul(np.array(self.message), self.public_key)
        return vector + self.error

    def decode(self, code):
        x_ = np.matmul(np.array(code), self.Q_matrix)
        u_ = self.LDPC.decode_by_gallager(x_)
        message = np.matmul(np.array(u_), self.S_matrix)
        return message
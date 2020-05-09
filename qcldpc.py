import random
import numpy as np
import matplotlib.pyplot as plt
from random import choice


def plot_parity_matrix(matrix):
    M = []
    for l in matrix:
        M.append([])
        for i in l:
            M[-1].append(int(i))
    plt.show()


class LDPC():

    def __init__(self, n0, r, w, m, t):
        self.n0 = n0
        self.r = r
        self.n = self.r * self.n0
        self.w = w
        self.m = m
        self.t = t
        self.R = (self.n0 - 1) / self.n0
        self.d = self.t * 2 + 1

        self.Q_matrix, Q_inv = self.gen_Q_matrix()

        self.H_matrix, self.G_matrix = self.get_generator_matrix()
        print("Generate parity check matrix")

        result = (np.matmul(self.H_matrix, self.G_matrix.transpose()) % 2).astype(int)
        if 1 in result:
            raise ValueError("Multiply H and G_transpose not equals 0")

        self.S_matrix, S_inv = self.gen_S_matrix()

        s_g = (np.matmul(S_inv, self.G_matrix) % 2).astype(int)
        print("Multiply S_inv and G")

        self.public_key = (np.matmul(s_g, Q_inv) % 2).astype(int)
        print("Multiply S_G and Q")

        print('ok')

    def get_generator_matrix(self):
        H0, H1, H_base = self.gen_qc_cyclic_matrix()
        H1_inv = (np.linalg.inv(H1) % 2).astype(int)
        H = (np.matmul(H1_inv, H0).transpose() % 2).astype(int)
        I = gen_identity_matrix(self.r)

        return H_base, np.concatenate((I, H), axis=1)

    def gen_Q_matrix(self):
        vector_1 = gen_random_vector_with_fixed_weight(self.n, self.m)
        Q_matrix = self.gen_cyclic_matrix(vector=vector_1)
        Q_inv = (np.linalg.inv(Q_matrix) % 2).astype(int)
        print("\nInverse Q")

        return Q_matrix, Q_inv

    def gen_S_matrix(self):
        S_matrix = gen_scrambling_matrix(self.r)
        S_inv = np.linalg.inv(S_matrix).astype(int)
        print("Inverse S")

        return S_matrix, S_inv

    def gen_cyclic_matrix(self, vector, row_count=None):
        block_size = len(vector)
        if row_count is not None:
            block_size = row_count
        check_matrix = [[] for _ in range(block_size)]
        for i, _ in enumerate(check_matrix):
            if row_count is not None and i == row_count:
                break
            check_matrix[i] = vector[-i:] + vector[:-i]

        return check_matrix

    def gen_qc_cyclic_matrix(self):
        h0 = gen_random_vector_with_fixed_weight(self.r, self.w // 2)
        h1 = gen_random_vector_with_fixed_weight(self.r, self.w // 2)
        H1 = self.gen_cyclic_matrix(h0)
        H0 = self.gen_cyclic_matrix(h1)

        return H0, H1, np.concatenate((H0, H1), axis=1)

    def decode_by_gallager(self, word, b=3, max_it=7):
        code = np.array(np.matmul(np.array(word), self.Q_matrix) % 2)
        iter = 0
        e = []
        while iter < max_it:
            max_upc = 0
            counters_upc = [0 for i in range(len(word))]
            syndrome = np.matmul(self.H_matrix, np.array(code).transpose()) % 2

            for i in range(0, self.r):
                if syndrome[i] == 1:
                    for j in range(self.n):
                        if self.H_matrix[i][j] == 1:
                            counters_upc[j] += 1

            for i in range(self.n):
                if max_upc > counters_upc[i]:
                    max_upc = counters_upc[i]

            for i in range(self.n):
                if counters_upc[i] > max_upc:
                    code[i] ^= 1

            e.append(np.array(counters_upc) % 2)
            temp_code = np.fmod(e[iter], code)
            temp_s = np.matmul(self.H_matrix, np.array(temp_code).transpose()) % 2
            s = np.matmul(self.H_matrix, np.array(code).transpose()) % 2

            unique, counts = np.unique(s, return_counts=True)
            print(dict(zip(unique, counts)))

            if 1 not in s:
                return code
            else:
                iter += 1
                b -= 1

        print("Decoding failure...")
        return [0 for _ in range(len(word))]


def gen_random_vector_with_fixed_weight(size, weight):
    random_range_size = range(size)
    range_items = random.sample(random_range_size, k=len(random_range_size))
    vector = [0 for _ in range(size)]
    for i in range_items[:weight]:
        hamming_weight = get_hamming_weight(vector)
        if hamming_weight == weight:
            break
        vector[i] = 1

    return vector


def gen_identity_matrix(size):
    return np.identity(size, dtype=int)


def get_hamming_weight(num):
    length = len(num)
    weight = 0
    for i in range(length):
        if num[i] & 1 == 1:
            weight += 1
    return weight


def gen_vector_with_weight(num):
    length = len(bin(num)) - 2
    weight = 0
    for i in range(length):
        if num & 1 == 1:
            weight += 1
        num >>= 1
    return weight


def gen_scrambling_matrix(length):
    size = range(length)
    check_matrix = [[0] * length for _ in size]
    positions = comb_choice(size, length)
    random.shuffle(positions)
    for i in range(length):
        check_matrix[i][positions[i]] = 1

    return check_matrix


def comb_choice(sample_list, n):
    sample = set()
    while len(sample) < n:
        position = choice(sample_list)
        if position not in sample:
            sample.add(position)
    return list(sample)

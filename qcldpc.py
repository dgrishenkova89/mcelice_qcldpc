import random
import numpy as np
import math

import matplotlib.pyplot as plt

from random import choice
from collections import defaultdict

def plot_parity_matrix(matrix):
    M = []
    for l in matrix:
        M.append([])
        for i in l:
            M[-1].append(int(i))
    plt.show()


class LDPC():

    def __init__(self, n0, r, w, t, m, error_prob):
        self.n0 = n0
        self.r = r
        self.n = self.r * self.n0
        self.w = w
        self.t = t
        self.m = m
        self.R = (self.n0 - 1) / self.n0
        self.error_prob = error_prob

        self.message = self.gen_random_message()
        self.error = gen_random_vector_with_fixed_weight(self.n, self.t // self.m)

        try:
            self.H_matrix, self.G_matrix = self.get_generator_matrix(self.r, self.w)
            print("Generate parity check matrix")

            vector_q = gen_random_vector_with_fixed_weight(self.n, self.m // 2)
            self.Q_matrix = self.gen_cyclic_matrix(vector_q)
            Q_inv = np.linalg.inv(self.Q_matrix).astype(int)
            print("Inverse Q")

            vector_s = gen_random_vector_with_fixed_weight(self.r, self.m)
            self.S_matrix = self.gen_cyclic_matrix(vector_s)
            self.S_inv = np.linalg.inv(self.S_matrix).astype(int)
            print("Inverse S")

            s_g = np.matmul(self.S_inv, self.G_matrix)
            print("Multiply S_inv and G")

            self.public_key = np.matmul(s_g, Q_inv)
            print("Multiply S_G and Q")
        except Exception:
            print(Exception)

        print('ok')

    def gen_random_message(self):
        message = np.array([random.choice([0, 1])
                            for i in range(self.r)])
        return message

    def get_generator_matrix(self, size, weight):
        H_base = self.gen_qc_cyclic_matrix(size, weight)
        H1_inv = np.linalg.inv(H_base[:,size:(size * 2)]).astype(int)

        H = np.matmul(H1_inv, H_base[:,0:size]).transpose()
        I = gen_identity_matrix(size)
        return H_base, np.concatenate((I, H), axis=1)

    def gen_cyclic_matrix(self, vector):
        block_size = len(vector)
        check_matrix = [[] for _ in range(block_size)]
        for i, _ in enumerate(check_matrix):
            check_matrix[i] = vector[i:] + vector[:i]

        return check_matrix

    def gen_qc_cyclic_matrix(self, size, weight):
        h0 = gen_random_vector_with_fixed_weight(size, weight // 2)
        h1 = gen_random_vector_with_fixed_weight(size, weight // 2)

        H0 = self.gen_cyclic_matrix(h0)
        H1 = self.gen_cyclic_matrix(h1)
        H = np.concatenate((H1, H0), axis=1)

        return H

    def decode_by_gallager(self, word, b=5, max_it=15):
        code = word
        iter = 0
        while iter < max_it:
            max_upc = 0
            counters_upc = [0 for i in range(len(word))]
            syndrome = np.matmul(self.H_matrix, np.array(code).transpose()) % 2

            for i in range(self.r):
                if syndrome[i] == 1:
                    for j in range(self.n):
                        if self.H_matrix[i][j] == 1:
                            counters_upc[j] += 1

            for i in range(self.n):
                if max_upc > counters_upc[i]:
                    max_upc = counters_upc[i]

            for i in range(self.n):
                if counters_upc[i] >= max_upc - b:
                    code[i] ^= 1

            s = np.matmul(self.H_matrix, np.array(code).transpose()) % 2

            unique, counts = np.unique(s, return_counts=True)
            print(dict(zip(unique, counts)))

            if s.all(0):
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
        if get_hamming_weight(vector) == weight:
            break
        vector[i] = 1

    return vector


def gen_identity_matrix(size):
    return np.identity(size, dtype=int)


def get_hamming_weight(num):
    length = len(num) - 2
    weight = 0
    for i in range(length):
        if num[i] & 1 == 1:
            weight += 1
        num[i] >>= 1
    return weight
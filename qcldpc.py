import random
import numpy as np

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
        self.H_matrix, self.H_base, self.G_matrix = get_generator_matrix(self.r, self.w)

        self.message = self.gen_random_message()
        self.error = self.gen_random_error()

        vector = gen_random_vector_with_fixed_weight(self.n, self.m)
        self.Q_matrix = gen_cyclic_matrix(get_cyclic_matrix(vector))

        self.S_matrix = gen_random_regular_row_matrix(self.m, self.r)
        s_g = np.matmul(np.linalg.inv(self.S_matrix), self.G_matrix)

        self.public_key = np.matmul(s_g, np.linalg.inv(self.Q_matrix))

        self.fill_nodes()

        print('ok')

    def gen_random_message(self):
        message = np.array([random.choice([0, 1])
                            for i in range(self.r)])
        return message

    def gen_random_error(self):
        error = []
        for i in range(self.n):
            if random.random() <= self.error_prob:
                error.append(1)
            else:
                error.append(0)
        return error

    def fill_nodes(self):
        self.check_nodes = [(i, 'c%d' % i)
                            for i in range(len(self.H_base))]
        self.variable_nodes = [(i, 'v%d' % i)
                               for i in range(len(self.H_base[0]))]

        self.check_adjacencies = defaultdict(list)
        self.variable_adjacencies = defaultdict(list)
        for check_node, line in zip(self.check_nodes, self.H_base):
            for variable_node, entry in zip(self.variable_nodes, line):
                if entry == 1:
                    self.check_adjacencies[check_node[0]].append(variable_node)
                    self.variable_adjacencies[variable_node[0]].append(check_node)

    def decode_by_gallager(self, word, b=3, max_it=15):
        variable_nodes = []
        for value in word:
            variable_nodes.append(value)

        message_index_for_check = {}
        for j, _ in self.variable_nodes:
            for i, _ in self.variable_adjacencies[j]:
                message_index_for_check[(j, i)] = variable_nodes[j]

        check_msg_var = {}
        for it in range(max_it):
            for i, _ in self.check_nodes:
                check_nodes_sum = 0
                for j, _ in self.check_adjacencies[i]:
                    check_nodes_sum += message_index_for_check[(j, i)]
                check_nodes_sum %= 2
                for j, _ in self.check_adjacencies[i]:
                    check_msg_var[(i, j)] = (check_nodes_sum - message_index_for_check[(j, i)]) % 2

            for j, _ in self.variable_nodes:
                d = 0
                for i, _ in self.variable_adjacencies[j]:
                    d += (check_msg_var[(i, j)] + variable_nodes[j]) % 2
                for q, _ in self.variable_adjacencies[j]:
                    delta = d - (check_msg_var[(q, j)] + variable_nodes[j]) % 2
                    if delta >= b:
                        message_index_for_check[(j, q)] = (variable_nodes[j] + 1) % 2
                        continue

                    message_index_for_check[(j, q)] = variable_nodes[j]

            message = []
            for j, _ in self.variable_nodes:
                msg_set = [check_msg_var[(i, j)] for i, _ in
                           self.variable_adjacencies[j]]
                if len(msg_set) % 2 == 0:
                    message.append(variable_nodes[j])
                if msg_set.count(1) > len(msg_set)/2:
                    message.append(1)
                else:
                    message.append(0)

            word = np.matrix(message)
            syndrome = (word * self.H_base.transpose())

            if 1 not in syndrome[0]:
                break
        return word


def gen_random_regular_row_matrix(row_weight, length):
    check_matrix = [[0] * length for _ in range(length)]
    possible_lines = range(length)
    for i in range(length):
        for j in comb_choice(possible_lines, row_weight):
            check_matrix[i][j] = 1

    return check_matrix


def comb_choice(sample_list, n):
    sample = set()
    while len(sample) < n:
        sample.add(choice(sample_list))
    return list(sample)


def get_generator_matrix(size, weight):
    H_base, H0, H1 = gen_qc_cyclic_matrix(size, weight)
    H = np.matmul(np.linalg.inv(H1), H0).transpose()
    I = gen_identity_matrix(size)
    return H, H_base, np.concatenate((I, H), axis=1)


def gen_cyclic_matrix(cyclic_matrix):
    block_size = len(cyclic_matrix)
    check_matrix = [[] for _ in range(block_size)]

    for i, _ in enumerate(check_matrix):
        check_matrix[i] = cyclic_matrix[i:] + cyclic_matrix[:i]

    return check_matrix


def get_cyclic_matrix(vector):
    size = range(len(vector))
    result = [0 for _ in size]
    for i in size:
        for j in zip(vector[-i:], vector[:-i]):
            result[i] ^= sum(j)
    return result


def gen_qc_cyclic_matrix(size, weight):
    h0 = gen_random_vector_with_fixed_weight(size, int(weight / 2))
    h1 = gen_random_vector_with_fixed_weight(size, int(weight / 2))

    H0 = gen_cyclic_matrix(get_cyclic_matrix(h0))
    H1 = gen_cyclic_matrix(get_cyclic_matrix(h1))
    H_base = np.concatenate((H1, H0))

    return H_base, H0, H1


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

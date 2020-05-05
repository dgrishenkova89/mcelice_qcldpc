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

        self.H_matrix, self.G_matrix = self.get_generator_matrix(self.r, self.w)
        print("Generate parity check matrix")

        self.message = self.gen_random_message()
        self.error = self.gen_random_error()

        try:
            vector = gen_random_vector_with_fixed_weight(self.n, self.m)
            self.Q_matrix = self.gen_cyclic_matrix(vector)
            Q_inv = np.linalg.inv(self.Q_matrix).astype(int)
            print("Inverse Q")
            self.S_matrix = gen_random_regular_row_matrix(self.m, self.r)
            S_inv = np.linalg.inv(self.S_matrix).astype(int)
            print("Inverse S")
            s_g = np.matmul(S_inv, self.G_matrix)
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

    def gen_random_error(self):
        error = []
        for i in range(self.n):
            if random.random() <= self.error_prob:
                error.append(1)
            else:
                error.append(0)
        return error

    def decode_by_gallager(self, word, b=3, max_it=15):
        code = word
        iter = 0
        while iter < max_it:
            max_upc = 0
            counters_upc = [0 for i in range(len(word))]
            syndrome = np.matmul(code, self.H_matrix.transpose()) % 2
            if 1 not in syndrome:
                break

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

            s = np.matmul(code, self.H_matrix.transpose()) % 2
            if 1 in s:
                return code
            else:
                iter += 1
                b -= 1

        return [0 for i in range(len(word))]


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


#def get_cyclic_matrix(vector):
#    size = range(len(vector))
#    result = [0 for _ in size]
#    result[0] = vector
#    for i in range(1, size):
#        for j in zip(vector[-i:], vector[:-i]):
#            result[i] = sum(j)
#    return result



#    def fill_nodes(self):
#        self.check_nodes = [(i, '%d' % i)
#                            for i in range(len(self.H_matrix))]
#        self.variable_nodes = [(i, '%d' % i)
#                               for i in range(len(self.H_matrix[0]))]
#        self.check_adjacencies = defaultdict(list)
#        self.variable_adjacencies = defaultdict(list)
#        for check_node, line in zip(self.check_nodes, self.H_matrix):
#           for variable_node, entry in zip(self.variable_nodes, line):
#               if entry == 1:
#                   self.check_adjacencies[check_node[0]].append(variable_node)
#                   self.variable_adjacencies[variable_node[0]].append(check_node)

#    def decode_by_gallager(self, word, b=3, max_it=15):
#        variable_nodes = []
#        for value in word:
 #           variable_nodes.append(value)

#        message_index_for_check = {}
#        for j, _ in self.variable_nodes:
#            for i, _ in self.variable_adjacencies[j]:
#                message_index_for_check[(j, i)] = variable_nodes[j]

#        check_msg_var = {}
#        for it in range(max_it):
#            for i, _ in self.check_nodes:
#                check_nodes_sum = 0
#                for j, _ in self.check_adjacencies[i]:
#                    check_nodes_sum += message_index_for_check[(j, i)]
#                check_nodes_sum %= 2
#                for j, _ in self.check_adjacencies[i]:
#                    check_msg_var[(i, j)] = (check_nodes_sum - message_index_for_check[(j, i)]) % 2

#            for j, _ in self.variable_nodes:
#                d = 0
#                for i, _ in self.variable_adjacencies[j]:
#                    d += (check_msg_var[(i, j)] + variable_nodes[j]) % 2
#                for q, _ in self.variable_adjacencies[j]:
#                    delta = d - (check_msg_var[(q, j)] + variable_nodes[j]) % 2
#                    if delta >= b:
#                        message_index_for_check[(j, q)] = (variable_nodes[j] + 1) % 2
#                        continue

#                    message_index_for_check[(j, q)] = variable_nodes[j]

#            message = []
#            for j, _ in self.variable_nodes:
#                msg_set = [check_msg_var[(i, j)] for i, _ in
#                           self.variable_adjacencies[j]]
#                if len(msg_set) % 2 == 0:
#                    message.append(variable_nodes[j])
#                if msg_set.count(1) > len(msg_set)/2:
#                    message.append(1)
#                else:
#                    message.append(0)

#            word = np.matrix(message)
#            syndrome = (word * self.H_matrix.transpose())

#            if 1 in syndrome[0]:
#                break
#        return word
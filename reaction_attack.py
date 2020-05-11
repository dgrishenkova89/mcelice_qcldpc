from __future__ import division, print_function

from mcelice_qcldpc.qcldpc import gen_qc_cyclic_matrix, get_hamming_weight
from random import choice, shuffle
from collections import defaultdict

def get_spectrum_dist(vector):
    r = len(vector)
    ones_pos = [i for i, v in enumerate(vector) if v == 1]
    spectrum = defaultdict(int)

    for i, vi in enumerate(ones_pos):
        for j, vj in enumerate(ones_pos[i + 1:]):
            d = min(vj - vi, r - vj + vi)
            spectrum[d] += 1
    return spectrum


def min_dist(pos1, pos2, length):
    return min(abs(pos2 - pos1), length - abs(pos2 - pos1))


class reaction_attack:
    def __init__(self, n0, r):
        self.n0 = n0
        self.r = r
        self.n = self.r * self.n0

    def key_recovery(self, M):
        for i in range(M):
            size = self.r // 2
            w = choice(range(100))
            H0, H1, B = gen_qc_cyclic_matrix(self.r, w)

            s0 = get_spectrum_dist(H0[0])
            s1 = get_spectrum_dist(H1[0])

            list_out_s0 = [i for i in range(1, self.r // 2 + 1) if i not in s0]
            list_out_s1 = [i for i in range(1, self.r // 2 + 1) if i not in s1]

            if len(list_out_s0) == 0 or list(list_out_s1) == 0:
                continue

            shuffle(list_out_s0)
            shuffle(list_out_s1)

            out_s0 = set(list_out_s0[:size])
            out_s1 = set(list_out_s1[:size])

            d0 = min(s0)
            d1 = min(s1)

            h1_rec = self.restore_message_by_spectrum(out_s0, out_s1, d0, d1, B, w)

            if h1_rec != -1:
                return h1_rec


    def restore_message_by_spectrum(self, out_s0, out_s1, d0, d1, B, weight):
        partial_h0 = [0, d0]
        h0_zero_pos = [i for i in range(self.r) if any([min_dist(i, j, self.r) in out_s0 for j in partial_h0 if i != j])]

        h1_poss_ones = []
        for i in range(self.r):
            if min_dist(i, 0, self.r) not in out_s1 and not (d1 and min_dist(i, d1, self.r) in out_s1):
                h1_poss_ones.append(i)

        B_Z1 = [B[i] for i in list(h1_poss_ones)]
        for one_pos in range(len(B_Z1)):
            cols = [(one_pos + i) % self.r for i in h0_zero_pos]
            z1 = [B[one_pos][i] for i in cols]
            w = get_hamming_weight(B_Z1[one_pos])
            if w <= weight:
                return B_Z1[one_pos]
        return -1
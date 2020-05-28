from __future__ import division, print_function

import pandas as pd
import seaborn as sns
import mcelice_qcldpc.McElieceSystem as mc
import matplotlib.pyplot as plt
import numpy as np

from mcelice_qcldpc.qcldpc import gen_qc_cyclic_matrix, get_hamming_weight
from random import choice, shuffle
from collections import defaultdict


def histplot(spec, probs):

    df = pd.DataFrame()

    sorted_ks = sorted(probs.keys(), key=lambda k: probs[k])
    df['x'] = probs.keys()
    df['y'] = sorted(probs.values())

    df['in'] = [dist in spec for dist in sorted_ks]

    def color(x):
        if x:
            return 'green'
        return 'blue'

    sns.barplot(data=df, x='x', y='y', palette=[color(v) for v in df['in']])

def plot_disp(spec, probs):

    df = pd.DataFrame()

    df['x'] = sorted(probs.keys(), key=lambda x: probs[x])
    df['y'] = sorted(probs.values())
    ind = lambda x: 1 if x == True else 0
    df['hue'] = [ind(k in spec) for k in probs]

    sns.set_style("ticks")

    sns.lmplot('x', 'y',
               data=df,
               fit_reg=False,
               hue="hue",
               scatter_kws={"marker": "D",
                            "s": 50})


def violinplot(spec, probs, figname='xxx_333.svg'):

    inprobs = [probs[i] for i in spec]
    outprobs = [probs[o] for o in probs if o not in spec]

    palette = [sns.xkcd_rgb["grey"], sns.xkcd_rgb["medium green"],
               sns.xkcd_rgb["pale red"]]
    sns.boxplot(data=[list(probs.values()), inprobs, outprobs],
                   palette=palette, linewidth=1)

    plt.ylabel(r'Taxa de falha', fontsize=20)
    plt.xlabel(r'', fontsize=20)
    plt.ylim(0)
    plt.axes().set_xticklabels([r'Todas', r"Muito em cumum", r"Pouco em comums \\ em comum"])
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


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
    def __init__(self, n0, r, mc_system):
        self.n0 = n0
        self.r = r
        self.n = self.r * self.n0
        self.mc_system = mc_system
        self.source_cipher = mc_system.encode()
        self.source_message = mc_system.message
        self.source_error_vector = mc_system.error_vector

    def key_recovery(self):
        size = self.r // 2
        w = choice(range(size // 2))
        m = choice(range(50))
        t = choice(range(size // 2))
        p0, p1, B, cipher_text, decode_success = self.spectrum_recovery(w, m, t)

        s0 = get_spectrum_dist(B[0][0:self.r])
        s1 = get_spectrum_dist(B[0][self.r:(self.r * 2)])

        if len(s0) == 0 or list(s1) == 0:
            return False, None, cipher_text, p0, p1, s0, s1, decode_success

        list_out_s0 = [i for i in range(1, self.r // 2 + 1) if i not in s0]
        list_out_s1 = [i for i in range(1, self.r // 2 + 1) if i not in s1]

        if len(list_out_s0) == 0 or list(list_out_s1) == 0:
            return False, None, cipher_text, p0, p1, s0, s1, decode_success

        shuffle(list_out_s0)
        shuffle(list_out_s1)

        out_s0 = set(list_out_s0[:size])
        out_s1 = set(list_out_s1[:size])

        d0 = min(s0)
        d1 = min(s1)

        h1_rec = self.restore_message_by_spectrum(out_s0, out_s1, d0, d1, B, w)

        if h1_rec is not None:
            return True, h1_rec, cipher_text, p0, p1, np.array(s0), np.array(s1), decode_success


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
        return None

    def spectrum_recovery(self, w, m, t):
        crypto = mc.McElieceSystem(self.n0, self.r, w, m, t)
        cipher_text = crypto.encode()
        message, success = self.mc_system.decode(cipher_text)

        s0 = get_spectrum_dist(crypto.error_vector[self.r:])
        s1 = get_spectrum_dist(crypto.error_vector[:self.r])

        a0 = [(i + int(success)) for i in range(self.r) if i in s0]
        b0 = [i for i in range(self.r) if i in s0]

        a1 = [(i + int(success)) for i in range(self.r) if i in s1]
        b1 = [i for i in range(self.r) if i in s1]

        p0 = [(a0[i] / b0[i]) for i in range(len(a0))]
        p1 = [(a1[i] / b1[i]) for i in range(len(a1))]

        return np.array(p0), np.array(p1), crypto.LDPC.H_matrix, cipher_text, success


if __name__ == "__main__":
    n, m, t, k = 64, 6, 4, 40
    GF2 = sg.GF(2)

    mceliece_system = McElieceSystem(k, m, t, scrambling=True)

    alice = PublicKeyHolder(*mceliece_system.get_public_key())
    bernie = PrivateKeyHolder(*mceliece_system.get_private_key())

    message = sg.Matrix([GF2.random_element() for i in range(k)])
    print('message\n', message.str())
    ciphertext = alice.encrypt(message)
    print('ciphertext\n', ciphertext.str())
    plaintext = bernie.decrypt(ciphertext)
    print('plaintext\n', plaintext.str())
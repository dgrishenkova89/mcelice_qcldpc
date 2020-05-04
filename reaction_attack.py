from __future__ import division, print_function

import random

from collections import defaultdict

BLOCK_SIZE = 4801


def get_spectrum_dense(vector):
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


def get_spectrum(vector, length):
    spectrum = defaultdict(int)
    for vi in vector:
        for vj in vector:
            if vi >= vj:
                continue
            d = min_dist(vi, vj, length)
            spectrum[d] += 1

    return dict(spectrum)


def get_code_spectrum(code):
    r = code.length / 2
    h0 = code.check_matrix[0][:r]

    return get_spectrum(h0, code.length)


class ReactionAttack:

    def restore_vector_from_spectrum(self, spectrum, vector, depth, weight, length, found):
        if depth == weight - 1:
            found.append(list(vector))
            return vector

        for i in spectrum:
            pos = vector[-1] + i
            if spectrum[i] < 1 or pos in vector:
                continue

            error = min(30, len(vector))

            selected_val = random.sample(vector, min(30, len(vector)))
            for j, position in enumerate(selected_val):
                k = min_dist(position, pos, length)
                if spectrum.get(k, 0) < 1:
                    error = j
                    break
                spectrum[k] -= 1
            else:
                vector.append(pos % length)
                restore_vector = self.restore_vector_from_spectrum(spectrum, vector, depth + 1, weight, length, found)
                const = [0] * length
                for k in restore_vector:
                    const[k] = 1
                vector.pop()

            for position in selected_val[:error]:
                k = min_dist(position, pos, length)
                spectrum[k] += 1

        return vector

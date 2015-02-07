#! /usr/bin/env python
# coding:utf-8


import numpy as np
from logging import getLogger


class Corpus():
    def __init__(self, filename, dictionary):
        self.filename = filename
        self.dictionary = dictionary

    def __iter__(self):
        for line in (line.strip() for line in open(self.filename)):
            yield [self.dictionary[word] for word in line.split()]


def mkdictionary(xs: [[str]]) -> {str: int}:
    """
    Return dictionary: {str: int}
    """
    _id = 0
    dic = dict()
    for line in xs:
        for word in line:
            if word not in dic:
                dic[word] = _id
                _id += 1
    return dic


def file2dic(filename):
    return mkdictionary(
        line.strip().split() for line in open(filename)
    )


class LDA:
    def __init__(
        self,
        dictionary,
        num_z: int=2,
        logger=None
    ):
        self.dictionary = dictionary
        self.num_z = num_z

        self.logger = logger if logger else getLogger(__file__)

    def train(
        self,
        corpus,
        alpha: float=2.,
        beta: float=2.,
    ):

        # count documents and words
        num_d = 0
        w_set = set()
        for doc in corpus:
            num_d += 1
            w_set = w_set.union(set(doc))
        num_w = len(w_set)

        self.logger.info("train model: num_d={}, num_w={}, num_z={}".format(
            num_d, num_w, self.num_z))

        n_m = np.zeros(num_d)
        n_mz = np.zeros((num_d, self.num_z))
        n_wz = np.zeros((num_w, self.num_z))
        n_z = np.zeros(self.num_z)

        zs = []

        for m, doc in enumerate(corpus):
            n_m[m] = len(doc)
            z_m = np.random.randint(self.num_z, size=len(doc))
            zs.append(z_m)
            for w, z in zip(doc, z_m):
                n_mz[m, z] += 1
                n_wz[w, z] += 1
                n_z[z] += 1

        # sampling z
        for i in range(100):
            for m, doc in enumerate(corpus):
                for n, w in enumerate(doc):
                    # set current topic
                    z_mn = zs[m][n]

                    n_mz[m, z_mn] -= 1
                    n_wz[w, z_mn] -= 1
                    n_z[z_mn] -= 1
                    p_z = (n_mz[m] + alpha) * (n_wz[w] + beta) \
                        / (n_z + num_w * beta)
                    # normalize p_z
                    p_z = p_z / p_z.sum()

                    new_z_mn = np.random.multinomial(1, p_z).argmax()
                    # set new topic
                    zs[m][n] = new_z_mn
                    n_mz[m, new_z_mn] += 1
                    n_wz[w, new_z_mn] += 1
                    n_z[new_z_mn] += 1

        # define
        #   P(w|z)
        #   P(z|m)
        self.p_wz = (n_wz + beta) / (n_z + num_w * beta)
        self.p_zm = (n_mz.T + alpha) / (n_m + self.num_z * alpha)

        # define nums
        self.num_d = num_d
        self.num_w = num_w
        self.logger.debug(self.p_wz)
        self.logger.debug(self.p_zm)

    def save(
        self,
        filename: str,
    ):
        with open(filename, "w") as fd:
            print(
                "{} {} {}".format(self.num_d, self.num_w, self.num_z),
                file=fd
            )
            print("", file=fd)
            for w in range(self.num_w):
                for z in range(self.num_z):
                    print(
                        "{} {}\t{}".format(w, z, self.p_wz[w][z]),
                        file=fd
                    )
            print("", file=fd)
            for z in range(self.num_z):
                for m in range(self.num_d):
                    print(
                        "{} {}\t{}".format(z, m, self.p_zm[z][m]),
                        file=fd
                    )

        self.logger.info(
            "save model in {}: num_d={}, num_w={}, num_z={}".format(
                filename,
                self.num_d, self.num_w, self.num_z
            )
        )

    def load(
        self,
        filename,
    ):

        with open(filename) as fd:
            for line in (_.strip() for _ in fd):
                if line == "":
                    break
                else:
                    num_d, num_w, num_z = [int(num) for num in line.split()]

            # define instance variables
            self.num_d = num_d
            self.num_w = num_w
            self.num_z = num_z

            p_wz = np.zeros((num_w, num_z))
            p_zm = np.zeros((num_z, num_d))

            for array in [p_wz, p_zm]:
                for line in (_.strip() for _ in fd):
                    if line == "":
                        break
                    else:
                        indexes, prob = line.split("\t")
                        prob = float(prob)
                        x, y = [int(i) for i in indexes.split()]
                        array[x][y] = prob

            # define instance variables
            self.p_wz = p_wz
            self.p_zm = p_zm

            self.logger.info(
                "load model from {}: num_d={}, num_w={}, num_z={}".format(
                    filename,
                    num_d, num_w, num_z
                )
            )

    def show_prob(self):
        """
        show topic vectors
        """
        print(self.p_zm.T)


if __name__ == '__main__':

    from logging import basicConfig, INFO, DEBUG
    import os
    import argparse

    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="filename"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show DEBUG log"
    )
    parser.add_argument(
        "-n", "--num_z",
        type=int,
        nargs="?",
        default=2,
        help="show DEBUG log"
    )
    args = parser.parse_args()

    # logger
    basicConfig(
        level=DEBUG if args.verbose else INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    modelname = "{}.model".format(
        os.path.splitext(os.path.basename(args.filename))[0]
    )
    dictionary = file2dic(args.filename)
    corpus = Corpus(args.filename, dictionary)

    lda = LDA(dictionary, num_z=args.num_z)
    lda.train(corpus)
    lda.save(modelname)
    lda.load(modelname)
    lda.show_prob()

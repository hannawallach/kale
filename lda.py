from numpy import ones, tile, zeros
from numpy.random import seed
from scipy.special import gammaln, psi
from corpus import *
from iterview import iterview
from math_utils import sample
from plotting_utils import InteractivePlot


class LDA(object):

    def __init__(self, corpus,
                 num_topics=100,
                 num_gibbs_itns=250,
                 num_hyperparam_itns=5):
        """
        Initializes LDA and runs Gibbs sampling.

        Arguments:

        corpus -- corpus of documents

        Keyword arguments:

        num_topics -- number of topics
        num_gibbs_itns -- number of Gibbs sampling iterations
        num_hyperparam_itns --
        """

        assert T >= 1 and num_gibbs_itns > 0 and num_hyperparam_itns >= 0

        self.corpus = corpus

        self.T = T = num_topics

        self.D = D = len(corpus)
        self.V = V = len(corpus.vocab)

        self.alpha, self.m = alpha, m = 0.1 * T, ones(T) / T
        self.beta, self.n = beta, n = 0.01 * V, ones(V) / V

        # precompute the product of the concentration parameter and
        # mean for each Dirichlet prior

        self.alpha_m = alpha_m = alpha * m
        self.beta_n = beta_n = beta * n

        self.D_sum_gammaln_alpha_m = D * gammaln(alpha_m).sum()
        self.D_gammaln_alpha = D * gammaln(alpha)
        self.T_sum_gammaln_beta_n = T * gammaln(beta_n).sum()
        self.T_gammaln_beta = T * gammaln(beta)

        # allocate space for N_{t|d} + alpha * m_t, N_d + alpha,
        # N_{v|t} + beta * n_v, N_t + beta

        self.Ntd_plus_alpha_m = tile(alpha_m, (D, 1))
        self.Nd_plus_alpha = alpha * ones(D)
        self.Nvt_plus_beta_n = tile(beta_n, (T, 1)).T
        self.Nt_plus_beta = beta * ones(T)

        # allocate space for token--topic assignments

        self.z = z = []

        for doc in corpus:
            z.append(zeros(len(doc), dtype=int))

        self.gibbs_sampling(num_gibbs_itns, num_hyperparam_itns)

    def log_evidence_corpus_and_z(self):
        """
        Returns the log evidence for the corpus and the current sample
        of token--topic assignments: log(P(corpus, z)).
        """

        return (self.T_gammaln_beta
                - self.T_sum_gammaln_beta_n
                + gammaln(self.Nvt_plus_beta_n).sum()
                - gammaln(self.Nt_plus_beta).sum()
                + self.D_gammaln_alpha
                - self.D_sum_gammaln_alpha_m
                + gammaln(self.Ntd_plus_alpha_m).sum()
                - gammaln(self.Nd_plus_alpha).sum())

    def print_top_types(self):
        """
        Computes an approximation to the mean of the posterior
        distribution over topics using the current sample of
        token--topic assignments and prints the most probable word
        types for each topic according to this posterior distribution.
        """

        mean = self.Nvt_plus_beta_n / tile(self.Nt_plus_beta, (self.V, 1))

        for t in xrange(self.T):
            print '*', self.corpus.vocab.top_types(mean[:, t])

    def gibbs_sampling(self, num_gibbs_itns, num_hyperparam_itns):
        """
        Uses Gibbs sampling to draw multiple samples from the
        posterior distribution over token--topic assignments.

        Keyword arguments:

        num_gibbs_itns -- number of Gibbs sampling iterations
        num_hyperparam_itns --
        """

        print 'Initialization:'

        self.gibbs_iteration(init=True)

        log_evidence = self.log_evidence_corpus_and_z()

        plt = InteractivePlot('iteration', 'log(P(corpus, z))')
        plt.update(0, log_evidence)

        print
        print 'log(P(corpus, z)): %s' % log_evidence
        print 'alpha, beta: %s, %s\n' % (self.alpha, self.beta)
        self.print_top_types()

        for itn in xrange(1, num_gibbs_itns + 1):

            print
            print 'Iteration %s:' % itn

            self.gibbs_iteration()

            if num_hyperparam_itns > 0:
                self.optimize_alpha_m(num_hyperparam_itns)
                self.optimize_beta(num_hyperparam_itns)

            log_evidence = self.log_evidence_corpus_and_z()

            plt.update(itn, log_evidence)

            print
            print 'log(P(corpus, z): %s' % log_evidence
            print 'alpha, beta: %s, %s\n' % (self.alpha, self.beta)
            self.print_top_types()

    def gibbs_iteration(self, init=False):
        """
        Uses Gibbs sampling to draw a single sample from the posterior
        distribution over token--topic assignments.

        Keyword arguments:

        init -- whether to initialize token--topic assignments
        """

        corpus = self.corpus

        Ntd_plus_alpha_m = self.Ntd_plus_alpha_m
        Nd_plus_alpha = self.Nd_plus_alpha
        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        z = self.z

        for d, (doc, zd) in enumerate(iterview(zip(corpus, z), inc=200)):
            for n, (v, t) in enumerate(zip(doc.tokens, zd)):

                if not init:
                    Ntd_plus_alpha_m[d, t] -= 1
                    Nvt_plus_beta_n[v, t] -= 1
                    Nt_plus_beta[t] -= 1
                else:
                    Nd_plus_alpha[d] += 1

                t = sample((Nvt_plus_beta_n[v, :] / Nt_plus_beta)
                           * Ntd_plus_alpha_m[d, :])

                Ntd_plus_alpha_m[d, t] +=1
                Nvt_plus_beta_n[v, t] += 1
                Nt_plus_beta[t] += 1

                zd[n] = t

    def optimize_alpha_m(self, num_itns):
        """
        Jointly optimizes hyperparameters alpha and m using the
        current sample of token--topic assignments.

        Keyword arguments:

        num_itns -- number of optimization iterations
        """

        D = self.D

        alpha, alpha_m = self.alpha, self.alpha_m

        Ntd = self.Ntd_plus_alpha_m - tile(alpha_m, (D, 1))
        Nd = self.Nd_plus_alpha - alpha

        new_alpha, new_alpha_m = alpha, alpha_m.copy()

        for itn in xrange(1, num_itns + 1):

            new_alpha_m *= ((psi(Ntd + tile(new_alpha_m, (D, 1)))
                             - psi(tile(new_alpha_m, (D, 1)))).sum(axis=0)
                            / (psi(Nd + new_alpha) - psi(new_alpha)).sum())

            new_alpha = new_alpha_m.sum()

        self.alpha, self.alpha_m = new_alpha, new_alpha_m

        self.D_gammaln_alpha = D * gammaln(new_alpha)
        self.D_sum_gammaln_alpha_m = D * gammaln(new_alpha_m).sum()

        self.Ntd_plus_alpha_m = Ntd + tile(new_alpha_m, (D, 1))
        self.Nd_plus_alpha = Nd + new_alpha

    def optimize_beta(self, num_itns):
        """
        Optimizes hyperparameter beta using the current sample of
        token--topic assignments; n is assumed to be uniform.

        Keyword arguments:

        num_itns -- number of optimization iterations
        """

        T, V = self.T, self.V

        beta, n, beta_n = self.beta, self.n, self.beta_n

        Nvt = self.Nvt_plus_beta_n - tile(beta_n, (T, 1)).T
        Nt = self.Nt_plus_beta - beta

        new_beta, new_beta_n = beta, beta_n.copy()

        for itn in xrange(1, num_itns + 1):

            new_beta *= ((psi(Nvt + tile(new_beta_n, (T, 1)).T)
                          - psi(tile(new_beta_n, (T, 1)).T)).sum()
                         / (V * (psi(Nt + new_beta) - psi(new_beta)).sum()))

            new_beta_n = new_beta * n

        self.beta, self.beta_n = new_beta, new_beta_n

        self.T_gammaln_beta = T * gammaln(new_beta)
        self.T_sum_gammaln_beta_n = T * gammaln(new_beta_n).sum()

        self.Nvt_plus_beta_n = Nvt + tile(new_beta_n, (T, 1)).T
        self.Nt_plus_beta = Nt + new_beta

import cPickle as pickle
from csv import reader
from itertools import islice
from numpy import asarray, bincount, log, zeros
from sys import stderr
from text_utils import create_stopword_list, tokenize
from vocabulary import Vocabulary


class Document(object):

    def __init__(self, corpus, name, tokens):
        """
        Initializes this document's containing corpus, name, and
        tokens (represented as an array of type indices).

        Arguments:

        corpus --
        name --
        tokens --
        """

        self.corpus = corpus

        self.name = name
        self.tokens = asarray(tokens)

    def __len__(self):
        """
        Returns the number of tokens in this document.
        """

        return len(self.tokens)

    def plaintext(self):
        """
        Returns a string representation of this document.
        """

        return ' '.join([self.corpus.vocab.lookup(x) for x in self.tokens])

    def Nv(self):
        """
        Returns an array of length V containing the number of times
        each type occurs as a token in this document.
        """

        Nv = zeros(len(self.corpus.vocab), dtype=int)
        Nv[:self.tokens.max() + 1] = bincount(self.tokens)

        return Nv

    def tfidf(self):
        """
        Returns the TF-IDF representation for the document.
        """

        return self.Nv() * self.corpus.idf()


class Corpus(object):
    """
    Corpus of documents.
    """

    def __init__(self, documents=None, vocab=None):
        """
        Initializes this corpus' list of documents and vocabulary.
        """

        if documents is None:
            self.documents = []
        else:
            self.documents = documents

        if vocab is None:
            self.vocab = Vocabulary()
        else:
            self.vocab = vocab

        self._idf = None

    def __getitem__(self, i):
        """
        Returns the document corresponding to the specified index.

        i -- index for which to return document
        """

        return self.documents[i]

    def __getslice__(self, i, j):
        """
        Returns a new corpus, consisting of a subset of this corpus's
        documents, corresponding to the specified indices.

        Arguments:

        i -- start index (inclusive)
        j -- end index (exclusive)
        """
        return Corpus(self.documents[i:j], self.vocab)

    def __iter__(self):
        """
        Returns an iterator over all documents in this corpus.
        """

        return iter(self.documents)

    def __len__(self):
        """
        Returns the number of documents in this corpus.
        """

        return len(self.documents)

    def add(self, name, tokens):
        """
        Adds a new document to this corpus.

        Arguments:

        name -- document name
        tokens -- list of tokens (strings)
        """

        tokens = [self.vocab[x] for x in tokens]
        self.documents.append(Document(self, name, tokens))

        self._idf = None # cached IDF is now stale

    def plaintext(self):
        """
        Returns a string representation of this corpus.
        """

        for doc in self.documents:
            print '%s: %s' % (doc.name, doc.plaintext())

    @classmethod
    def load(cls, f):
        """
        Unpickles a corpus from the specified file.

        Arguments:

        f -- (name of) file
        """

        return pickle.load(get_file(f));

    def save(self, f):
        """
        Pickles this corpus to the specified file.

        Arguments:

        f -- (name of) file
        """

        if isinstance(f, basestring):
            f = open(f, 'wb')

        pickle.dump(self, f)

    @classmethod
    def from_csv(cls, f,
                 delimiter=',',
                 quotechar='"',
                 header_rows=0,
                 stopword_f=None):
        """
        Returns a new corpus of documents constructed from the
        specified CSV file. Each document represents a single row in
        the CSV: the first field is used as the document's name; the
        last field is used as the document's contents.

        Arguments:

        f -- (name of) CSV file

        Keyword arguments:

        delimiter --
        header_rows -- number of header rows to skip
        stopword_f -- (name of) file containing stopwords to remove
        """

        stopwords = create_stopword_list(stopword_f)

        corpus = cls()

        for fields in load_csv(f, delimiter, quotechar, header_rows):
            corpus.add(fields[0], tokenize(fields[-1], stopwords))

        print >> stderr, '# documents = ', len(corpus)
        print >> stderr, '# tokens = ', corpus.N()
        print >> stderr, '# types = ', len(corpus.vocab)

        return corpus

    def Nv(self):
        """
        Returns an array of length V containing the number of times
        each type occurs as a token in this corpus.
        """

        Nv = zeros(len(self.vocab), dtype=int)

        for doc in self.documents:
            Nv += doc.Nv()

        return Nv

    def Nd(self):
        """
        Returns an array containing the number of tokens in each
        document in this corpus.
        """

        Nd = zeros(len(self.documents), dtype=int)

        for d, doc in enumerate(self.documents):
            Nd[d] = len(doc)

        return Nd

    def N(self):
        """
        Returns the number of tokens in this corpus.
        """

        return sum(len(doc) for doc in self.documents)

    def Dv(self):
        """
        Returns an array of length V containing the number of
        documents in which each type occurs in this corpus.
        """

        Dv = zeros(len(self.vocab), dtype=int)

        for doc in self.documents:
            Dv[doc.tokens] += 1

        return Dv

    def idf(self, force=False):
        """
        Returns the IDF representation for this corpus.

        Keyword arguments:

        force -- whether to force recomputation
        """

        if self._idf is None or force:
            self._idf = log(len(self.documents)) - log(self.Dv())

        return self._idf

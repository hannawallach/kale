from numpy import argsort


class Vocabulary(object):
    """
    Bijective mapping from unique word types (strings) to integer
    indices 0 ... V-1, where V is this vocabulary's size.
    """

    def __init__(self):
        """
        Initializes this vocabulary's forward mapping from types to
        indices and reverse mapping from indices to types.
        """

        self._forward = {}
        self._reverse = {}

        self._growing = True

        self._idx = 0

    def stop_growth(self):
        """
        Stop vocabulary growth.
        """

        self._growing = False

    def lookup(self, i):
        """
        Returns the type represented by the specified index.

        Arguments:

        i -- index for which to return the type
        """

        assert isinstance(i, int)

        return self._reverse[i]

    def plaintext(self):
        """
        Returns a string representation of this vocabulary.
        """

        contents = self._reverse.items()
        contents.sort(key=lambda x: x[0])

        return '\n'.join('%s\t%s' % (i, s) for i, s in contents)

    def top_types(self, weights, num=10):
        """
        Returns a string representation of the highest-weighted types
        according to the specified weights.

        Arguments:

        weights -- set of weights

        Keyword arguments:

        num -- number of top types
        """

        assert len(weights) == len(self._forward)

        top_types = map(self.lookup, argsort(weights))

        return ' '.join(top_types[-num:][::-1])

    def __contains__(self, s):
        """
        Returns either True or False, indicating whether this
        vocabulary contains the specified type.

        Arguments:

        s -- type
        """

        assert isinstance(s, basestring)

        return s in self._forward

    def __getitem__(self, s):
        """
        Returns the index that represents the specified type. If this
        vocabulary does not contain the specified type and if
        vocabulary growth has not been stopped, create a new
        type--index mapping and return the index.

        Arguments:

        s -- type
        """

        try:
            return self._forward[s]
        except KeyError:

            if not isinstance(s, basestring):
                raise ValueError('Invalid key (%s): must be a string.' % (s,))

            if not self._growing:
                return None

            i = self._forward[s] = self._idx
            self._reverse[i] = s
            self._idx += 1

            return i

    add = __getitem__

    def __iter__(self):
        """
        Returns an iterator over all types in this vocabulary.
        """

        for i in xrange(len(self)):
            yield self._reverse[i]

    def __len__(self):
        """
        Returns the number of type--index mappings in this vocabulary.
        """

        return len(self._forward)

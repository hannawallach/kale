import re
from csv import reader
from itertools import islice
from json import loads
from urllib2 import urlopen


def get_file(f):
    """
    Open a file for reading.

    Arguments --

    f -- (name of) file
    """

    if isinstance(f, file):
        return f
    elif isinstance(f, basestring):
        if f[:7] == 'http://':
            return urlopen(f)
        else:
            return open(f, 'r')
    else:
        raise TypeError('Invalid type: cannot open a %s.' % type(f).__name__)


def create_stopword_list(f):
    """
    Returns a set of stopwords.

    Arguments:

    f -- list of stopwords or (name of) file containing stopwords
    """

    if f is None:
        return set()

    return set(word.strip() for word in get_file(f))


def tokenize(text, stopwords=set()):
    """
    Returns a list of lowercase tokens corresponding to the specified
    string with stopwords (if any) removed.

    Arguments:

    text -- string to tokenize

    Keyword arguments:

    stopwords -- set of stopwords to remove
    """

    tokens = re.findall('[a-z]+', text.lower())

    return [x for x in tokens if x not in stopwords]


def load_csv(f, delimiter=',', quotechar='"', header_rows=0):
    """
    Loads a CSV field and yields one row (as a list) at a time.

    Arguments:

    f -- (name of) CSV file

    Keyword arguments:

    delimiter --
    quotechar --
    header_rows --
    """

    for fields in islice(reader(get_file(f),
                                delimiter=delimiter, quotechar=quotechar),
                         header_rows, None):
        yield fields


def load_json(f):
    """
    Loads a JSON file and yields one row (as a dict) at a time.

    Arguments:

    f -- (name of) JSON file

    Keyword arguments:

    delimiter --
    header_rows --
    """

    for row in get_file(f):
        try:
            yield loads(row)
        except ValueError as e:
            pass

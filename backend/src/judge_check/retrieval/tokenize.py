"""The single tokenizer used for BM25, everywhere.

Lowercase, then Unicode word characters (letters, digits, underscore). No stemming and no
stopword list: IDF already down-weights frequent words. And because this function is the
only tokenizer, the in-memory and the SQL BM25 see identical terms, which is what makes
their scores comparable at all (D-020).
"""

import re

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())

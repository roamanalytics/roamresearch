#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import signal

__author__ = 'Nick Dingwall and Christopher Potts'


right_boundary_str = r"""
  (?:
  \s+(?:on|in|at|for|to|by|of)\s*
  )"""


date_time_stamp_re = r"""\[?\(\d+/\d+\s+\d+\s+\w+\)\-?\]?\-?"""


price_re = r"""[\$₤£¥€]\s*\d+[\d.,]+\d+"""


# phrase_re = ur"""(?:[a-zA-Z'\s+&/]|(?:\d+\s)|(?:\d[a-z]))+"""

phrase_re = r"""
  (?:\s*
      (?:[a-zA-Z](?:[a-z\-A-Z'\s+&]*/?[a-z\-A-Z'\s+&]+)*[a-zA-Z]) # Standard phrases, working hard to treat // as a boundary
      |
      (?:\d+[a-z]+)         # Words like 2nd, 1st, 43rd
      |
      (?:%s)                # price_re abvove with spacing allowed around it
      |
      (?:\d+[\d.,]+\d+)     # Numbers that aren't prices (just like price_re but without currency mark)
  \s*)+""" % price_re


apostrophes = r"'\u0027\u02BC\u2019"


word_re = r"""
  (?:[a-z][a-z%s\-_]+[a-z])  # Words with apostrophes or dashes.
  |
  (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
  |
  (?:[\w_&]+)                    # Words without apostrophes or dashes.
  |
  (?:\.(?:\s*\.){1,})            # Ellipsis dots.
  |
  (?:\*{1,})                     # Asterisk runs.
  |
  (?:\S)                         # Everything else that isn't whitespace.
  """ % apostrophes


phone_number_re = r"""
  (?:
    (?:            # (international)
      \+?[01]
      [\-\s.]*
    )?
    (?:            # (area code)
      [\(]?
      \d{3}
      [\-\s.\)]*
    )?
    \d{3}          # exchange
    [\-\s.]*
    \d{4}          # base
  )"""


long_date_re = r"""
    (?:
      (?:0?[1-9]|1[12])      # month
      [\-\s/.\\]+
      (?:[012]?[0-9]|3[01])  # day
      (?:[\-\s/.\\]+
      \d{2,4})?           # year
      |
      (?:[012]?[0-9]|3[01])  # day
      [\-\s/.\\]+
      (?:0?[1-9]|1[12])      # month
      (?:[\-\s/.\\]+
      \d{2,4})?              # year
      |
      \d{2,4}                # year
      [\-\s/.\\]+
      (?:0?[1-9]|1[12])      # month
      [\-\s/.\\]+
      (?:[012]?[0-9]|3[01])  # day
      |
      (?:[012]?[0-9]|3[01])
      \s+
      (?:
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?
        |
        (?:January|February|March|April|May|June|July|August|September|October|November|December)
      )
      \s+
      \d{2,4}
      |
      (?:
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?
        |
        (?:January|February|March|April|May|June|July|August|September|October|November|December)
      )
      [\s,]+
      (?:[012]?[0-9]|3[01])
      [\s,]+
      \d{2,4}
   )"""


short_date_re = r"""
  (?:
    (?:
      (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?
      |
      (?:January|February|March|April|May|June|July|August|September|October|November|December)
    )
    \s+
    (?:[012]?[0-9]|3[01])
    |
    (?:[012]?[0-9]|3[01])
    \s+
    (?:
      (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?
      |
      (?:January|February|March|April|May|June|July|August|September|October|November|December)
    )
  )"""


urls = r"""
  (?:
    (?:
      [a-z][\w-]+:                            # URL protocol and colon
      (?:
        /{1,3}                                # 1-3 slashes
        |                                     # or
        [a-z0-9%]                             # Single letter or digit or '%' (Trying not to match e.g. "URI::Escape")
      )
      |                                       #  or
      www\d{0,3}[.]                           # "www.", "www1.", "www2." … "www999."
      |                                       #   or
      [a-z0-9.\-]+[.][a-z]{2,4}/              # looks like domain name followed by a slash
    )
    (?:                                       # One or more:
      [^\s()<>]+                              # Run of non-space, non-()<>
      |                                       #   or
      \((?:[^\s()<>]+|(?:\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
    )+
    (?:                                       # End with:
      \((?:[^\s()<>]+|(?:\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
      |                                       # or
      [^\s`!()\[\]{};:'".,<>?«»“”‘’]           # not a space or one of these punct chars
     )
  )"""


emoticons = r"""
  (?:                           # non-capturing group
    [<>]?                       # optional hat/brow
    [:;=8]                      # eyes
    [\-o\*\']?                  # optional nose
    [\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth
    |                           #### reverse orientation
    [\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth
    [\-o\*\']?                  # optional nose
    [:;=8]                      # eyes
    [<>]?                       # optional hat/brow
  )"""


tags = r"""<[^>]+>"""

hat_tip = r"[Hh]/[Tt]"

email = r"(?:[\w._%+-]+@[\w._%+-]+\.\w{2,})"

twitter_username = r"""(?:@[\w_]+)"""

twitter_hashtag = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    """Help us to control the tokenizer should it go out of control
    for some unknown reason related to the complexity of regex matching.
    """
    raise TimeoutException()


class WordOnlyTokenizer:
    """Seeks to identify only word-like things, including dates,
    numbers, and prices, but excluding URLs.

    Parameters
    ----------
    lower : bool
        Whether downcase all tokens.
    preserve_acronyms : bool
        If `True`, then tokens that return true for `isupper`
        will not be downcases even if `lower=True`.
    offsets : bool
        If `True`, then `tokenize` returns a list of (str, offset)
        tuples. If `False`, it returns just a list of str.
    remove_whitespace_tokens : bool
        If `True`, then tokens consisting of whitespace are removed.
        This is usually the expected behavior. If `False`, then they
        are kept, and we guarantee that `"".join(tokenize(s)) == s`
        modulo the case parameters to the tokenizer instance.
    """
    def __init__(self, lower=False, preserve_acronyms=True,
            offsets=False, remove_whitespace_tokens=True):
        self.offsets = offsets
        self.remove_whitespace_tokens = remove_whitespace_tokens
        self.regexstrings = (
            urls,
            emoticons,
            date_time_stamp_re,
            long_date_re,
            short_date_re,
            phone_number_re,
            twitter_username,
            twitter_hashtag,
            hat_tip,
            email,
            price_re,
            word_re)
        # The actual regex is used with findall to map strings to lists
        # of tokens:
        self.regex = re.compile(r"""(%s)""" % "|".join(self.regexstrings),
                                re.VERBOSE | re.I | re.UNICODE)
        self.lower = lower
        self.preserve_acronyms = preserve_acronyms

    def tokenize(self, s):
        """The tokenizing method.

        Parameters
        ----------
        s : str

        Returns
        -------
        list of str if `self.offsets=False`, else list of
        (str, int) tuples.
        """
        tokens = []
        # The Python regex module sometimes really blows up, taking
        # ages to tokenize. This will prevent that by limiting the
        # permitted tokenizing time based on the length of the input.
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        secs = int(round(1 + (len(s)/100000.0), 0))
        signal.alarm(secs)
        try:
            tokens = self.regex.findall(s)
        except TimeoutException:
            return []
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)
        # This cleanup is easier to do separately than to try to
        # get it just right with the core regex.
        tokens = [self._token_cleanup(s) for s in tokens]
        if self.offsets:
            total = 0
            indices = [0]
            for tok in tokens[:-1]:
                total += len(tok)
                indices.append(total)
            results = list(zip(tokens, indices))
            if self.remove_whitespace_tokens:
                results = [(w, i) for w, i in results if w.strip()]
            return results
        else:
            if self.remove_whitespace_tokens:
               tokens = [w for w in tokens if w.strip()]
        return tokens

    def _token_cleanup(self, s):
        if self.preserve_acronyms and s.isupper():
            return s
        elif self.lower:
            return s.lower()
        else:
            return s

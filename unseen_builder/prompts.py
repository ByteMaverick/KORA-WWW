"""The five prompts the builder sends: the naming model's three, the checks' two."""

ATTRIBUTES = """\
Name: {name}
Type: {kind}
List, as 'attribute: value' lines, four facts you know about this {kind}: its {attributes}. If you do not know it, write 'unknown'.
"""

GUESS_SOURCE = """\
The following facts describe a real-world entity whose name has been replaced. Facts:
{facts}
Which real-world entity is this? Answer with the name only.
"""

NAME = """\
You are naming a new, fictional {kind} for a research dataset.
Known attributes of this {kind}:
{attributes}
Propose {k} names that read as plausible real-world names for this kind of {kind}, fitting its era and setting.
Rules: each name must not be the name or title of any real {kind}; it must not be a pun on, a variant of, or
derived from a real one; no placeholders (no "John Doe", no "Example"); match the language and style the
attributes suggest. Make the names varied and unrelated to one another: no two may begin with the same letters,
and none may begin like these names already in use (avoid these stems): {avoid}.
Output exactly {k} names, one per line, nothing else.
"""

NAME_PHRASE = """\
You are describing a new, fictional {kind} for a research dataset.
Known attributes of this {kind}:
{attributes}
Propose {k} short descriptive phrases for it, the way a clinical terminology would list it (for example "Persistent
dry cough on exertion"): two to six words, in sentence case (a capital first letter, the rest lower case except for
proper nouns), no person's or place's name, no code.
Rules: describe it in your own words rather than repeat a standard term; no placeholders. Make the phrases varied
and unrelated to one another: no two may begin with the same word, and none may begin like these phrases already
in use (avoid these stems): {avoid}.
Output exactly {k} phrases, one per line, nothing else.
"""

NAME_SERIES = """\
You are naming a new, fictional recurring {kind} for a research dataset: the series one edition belongs to.
Known attributes of this edition:
{attributes}
Propose {k} series names that read as plausible real-world names for this kind of {kind}, fitting its era and
setting. Each name will be written {placement}, as in "{example}": give the name alone, with no year or number.
Rules: each name must not be the name of any real series or {kind}; it must not be a pun on, a variant of, or
derived from a real one; no placeholders (no "Example"). Make the names varied and unrelated to one another: no two
may begin with the same letters, and none may begin like these names already in use (avoid these stems): {avoid}.
Output exactly {k} names, one per line, nothing else.
"""

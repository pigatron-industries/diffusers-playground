import re

def findBetween(text, start, end, includetags=False):
    regex = f"\{start}.*?\{end}"
    token = re.search(regex, text)
    if token is not None:
        if includetags:
            return token.group()
        else:
            return token.group()[1:-1]
    else:
        return None


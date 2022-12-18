import re

def findBetween(text, start, end):
    regex = f"\{start}.*?\{end}"
    token = re.search(regex, text)
    if token is not None:
        return token.group()[1:-1]
    else:
        return None


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


def padNumber(number, length):
    number_str = str(number)
    number_of_zeros = length - len(number_str)
    return '0' * number_of_zeros + number_str
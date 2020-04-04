def alphanum(x):
    try:
        return int(x)
    except(ValueError):
        try:
            return float(x)
        except(ValueError):
            return x

def list_of_alphanums(string):

    return [alphanum(a) for a in string.split()]

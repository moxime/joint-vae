def make_list(o, default_for_all):

    if isinstance(o, str):
        o = [o]

    if o is None:
        return []
    if o and o[0] in ('all', 'default'):
        return type(default_for_all)(default_for_all)
    if o and o[0] == 'first':
        return [next(iter(default_for_all))]

    return o

def make_list(o, default_for_all):

    if o is None:
        return []
    if o in ('all', 'default'):
        return type(default_for_all)(default_for_all)
    if isinstance(o, str):
        return [o]

    return o

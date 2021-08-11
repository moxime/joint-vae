def make_list(o, default_for_all):

    if o is None:
        return []
    if o in ('all', 'default'):
        return default_for_all
    if isinstance(o, str):
        return [o]

    return o

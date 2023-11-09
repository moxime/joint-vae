class NoModelError(Exception):
    pass


class StateFileNotFoundError(FileNotFoundError):
    pass


class DeletedModelError(NoModelError):
    pass


class MissingKeys(Exception):
    def __str__(self):
        return 'MissingKeys({})'.format(', '.join(self.args[-1]))

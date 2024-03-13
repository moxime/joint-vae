from filelock import FileLock
import os


class Scheduler(object):

    def __init__(self, file_path):

        self.lock = FileLock(file_path + '.lock')
        self.file_path = file_path

        with self.lock:
            with open(self.file_path, 'r') as fp:
                for count, line in enumerate(fp):
                    pass

            self._num_of_arg_lines = count + 1

    def __len__(self):

        return self._num_of_arg_lines

    def __getitem__(self, i):

        if i >= self._num_of_arg_lines:
            raise IndexError
        with self.lock:

            with open(self.file_path, 'r') as fp:
                for count, line in enumerate(fp):
                    if count == i:
                        break

            return line.strip()

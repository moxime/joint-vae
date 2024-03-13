from filelock import FileLock
import os
import time
import logging


class Scheduler(object):

    def __init__(self, file_path=None, item=0, period=1):

        self.file_path = file_path
        self.period = period
        self.item = item

        if self.file_path:
            with open(self.file_path, 'r') as fp:
                for count, line in enumerate(fp):
                    if count == item:
                        break

                else:
                    raise IndexError

            self.line = line.strip()

    def start(self, block=False):
        if not self.file_path:
            return

        i = self.item
        if block:
            blocking_files = ['{}.{}'.format(self.file_path, _) for _ in range(i - 1, i - self.period - 1, -1)]
            logging.info('Waiting for {} to be deleted'.format(','.join(blocking_files)))

            while any(os.path.exists(f) for f in blocking_files):
                time.sleep(0.5)

            logging.info('{} deleted, going through'.format(','.join(blocking_files)))

        with open('{}.{}'.format(self.file_path, i), 'w') as fp:
            pass

    def stop(self):
        if not self.file_path:
            return
        try:
            os.remove('{}.{}'.format(self.file_path, self.item))
        except FileNotFoundError:
            pass

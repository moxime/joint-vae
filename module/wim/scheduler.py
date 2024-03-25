from filelock import FileLock
import os
import time
import logging


class Scheduler(object):

    def __init__(self, file_path=None, index=0):

        self.file_path = file_path
        self.index = index

        if self.file_path:
            try:
                with open(self.file_path, 'r') as fp:
                    for count, line in enumerate(fp):
                        if count == index:
                            break

                    else:
                        raise IndexError

                self.line = line.strip()
            except FileNotFoundError:
                logging.info('{} does not exist for scheduler'.format(self.file_path))

    def start(self, block=False):
        if not self.file_path:
            return

        current_idx = self.index
        if block:
            blocking_files = ['{}.{}'.format(self.file_path, _) for _ in block]
            logging.info('Waiting for {} to be deleted'.format(','.join(blocking_files)))

            t0 = time.time()
            while any(os.path.exists(f) for f in blocking_files):
                time.sleep(0.5)

            t1 = time.time()
            logging.info('{} deleted, going through (waited {:.1f}s)'.format(','.join(blocking_files), t1 - t0))

        with open('{}.{}'.format(self.file_path, current_idx), 'w') as fp:
            pass

    def stop(self):
        if not self.file_path:
            return
        try:
            os.remove('{}.{}'.format(self.file_path, self.index))
        except FileNotFoundError:
            pass

        time.sleep(1)

import signal
import logging



class SIGHandler:

    def __init__(self, *sigs):

        logging.debug('Registering signals ' + ' '.join([str(s) for s in sigs]))
        self.sig = 0
        self._sigs = sigs
        for s in sigs:
            signal.signal(s, self.handle)

    def handle(self, sig, _):

        self.sig = sig
        logging.warning(f'Catching signal {self}, crossing fingers')
    
    @classmethod
    def create(cls, *sigs):
        h = SIGHandler(*sigs)

        return h


    def __str__(self, *a, **kw):
        if self.sig:
            return signal.Signals(self.sig).name
        else:
            return 'handler for signals ' + ' '.join([str(s) for s in self._sigs])

import time
import numpy as np


def print_epoch(i, per_epoch, epoch, epochs, loss,
                snake='=>', blinker='o ', line_length=50,
                info='', end_of_epoch='\n'):

    steps = i * (line_length - len(info)) // per_epoch
    K = int(np.log10(epochs))+1
    Ki = int(np.log10(per_epoch))+1

    print('\r', end='')
    print(f'epoch {epoch+1:{K}d}/{epochs} ', end='')
    print('=' * steps + snake[i % len(snake)] +
          ' ' * (line_length - steps - len(info)) + f'{loss: .3e} {info}', end='')
    
    print(f' (batch {i+1:{Ki}d}/{per_epoch})',
          end = end_of_epoch if i == per_epoch - 1 else '')
    # print(f' {blinker[i%len(blinker)]}', end='')


class Time(float):

    def __str__(self, max=2):

        t = self
        i = 0

        if t == 0:
            return '0s'
        
        str = '-' if t < 0 else ''
        t = abs(t)
        
        d = int(t / (24 * 60 * 60))
        if d > 0:
            str += f'{d}d'
            i += 1
            t -= d * 24 * 60 * 60

        h = int(t / 3600)
        if h > 0:
            str += f'{h}h'
            i +=1
            t -= h * 3600

        m = int(t / 60)
        if m > 0 and i < max:
            str += f'{m}m'
            i += 1
            t -= m * 60

        s = int(t)
        if s > 0 and i < max:
            str += f'{s}s'
            i += 1
            t -= s

        m = int(1e3 * t)
        mu = int(1e6 * t - 1000 * m)
        
        if i < max - 1 and mu > 0:
            if m > 0:
                str += f'{m}ms{mu:03d}'
            else:
                str += f'{mu}us'
                
        elif i < max and m > 0:
            str += f'{m}ms'

        return str
        
    def __add__(self, t_):

        return Time(float(self) + float(t_))

    def __neg__(self):

        return Time(-float(self))
        
    def __sub__(self, t):

        return self + (-t)

    def __mul__(self, k):

        return Time(float(self) * k)

    def __truediv__(self, k):

        return Time(float(self) / k)
    
    def __format__(self, *a, **k):

        return str(self).__format__(*a, **k)


if __name__ == '__main__':
        
    epochs = 10
    per_epoch = 60

    print_epoch(per_epoch-1, per_epoch, -1, epochs, 0)

    for epoch in range(epochs):
        for i in range(per_epoch):

            l = np.random.randn()
            time.sleep(5)
            print_epoch(i, per_epoch, epoch, epochs, l, end_of_epoch='')



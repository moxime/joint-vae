import time
import numpy as np


def print_epoch(i, per_epoch, epoch, epochs, loss,
                snake='=<', blinker='o ', line_length=60, info=''):

    steps = i * (line_length - len(info)) // per_epoch
    K = int(np.log10(epochs))+1
    Ki = int(np.log10(per_epoch))+1

    print('\r', end='')
    print(f'epoch {epoch+1:{K}d}/{epochs} ', end='')
    print('=' * steps + snake[i % len(snake)] +
          ' ' * (line_length - steps - len(info)) + f'{loss: .3e} {info}', end='')
    if i == per_epoch - 1:
        print()
    else:
        pass
        print(f' (batch {i:{Ki}d}/{per_epoch})', end='')
        # print(f' {blinker[i%len(blinker)]}', end='')

if __name__ == '__main__':
        
    epochs = 10
    per_epoch = 600

    print_epoch(per_epoch-1, per_epoch, -1, epochs, 0)

    for epoch in range(epochs):
        for i in range(per_epoch):

            l = np.random.randn()
            time.sleep(0.05)
            print_epoch(i, per_epoch, epoch, epochs, l)



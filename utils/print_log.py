import time
import numpy as np


def print_epoch(i, per_epoch, epoch, epochs, loss,
                snake='=<', blinker='o ', line_length=60,
                info='', end_of_epoch='\n'):

    steps = i * (line_length - len(info)) // per_epoch
    K = int(np.log10(epochs))+1
    Ki = int(np.log10(per_epoch))+1

    print('\r', end='')
    print(f'epoch {epoch+1:{K}d}/{epochs} ', end='')
    print('=' * steps + snake[i % len(snake)] +
          ' ' * (line_length - steps - len(info)) + f'{loss: .3e} {info}', end='')
    
    print(f' (batch {i:{Ki}d}/{per_epoch})',
          end = end_of_epoch if i == per_epoch - 1 else '')
    # print(f' {blinker[i%len(blinker)]}', end='')

if __name__ == '__main__':
        
    epochs = 10
    per_epoch = 60

    print_epoch(per_epoch-1, per_epoch, -1, epochs, 0)

    for epoch in range(epochs):
        for i in range(per_epoch):

            l = np.random.randn()
            time.sleep(5)
            print_epoch(i, per_epoch, epoch, epochs, l, end_of_epoch='')



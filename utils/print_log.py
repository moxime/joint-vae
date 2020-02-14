epochs = 10
per_epoch = 30
import time
import numpy as np








def print_epoch(i, per_epoch, epoch, epochs, loss, snake='=/_==', blinker='o  '):
    K= int(np.log10(epochs))+1
    print('\r', end='')
    print(f'epoch {epoch+1:{K}d}/{epochs} ', end='')
    print('=' * i + snake[i%len(snake)] + ' ' * (per_epoch - i) + f'{loss: .3e}', end='')
    if i == per_epoch - 1:
        print()
    else:
        #print(f' ({i:{K}d}/{per_epoch})', end='')
        print(f' {blinker[i%len(blinker)]}', end='')

if __name__ == '__main__':
        
    print_epoch(per_epoch-1, per_epoch, -1, epochs, 0)

    for epoch in range(epochs):
        for i in range(per_epoch):

            l = np.random.randn()
            time.sleep(0.2)
            print_epoch(i, per_epoch, epoch, epochs, l)



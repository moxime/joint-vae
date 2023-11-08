from utils.save_load import _collect_thoroughly_models

nets = []

_collect_thoroughly_models('jobs/fashion', nets)

nets = sum(nets, [])

for n in nets:

    n_ = n['net']
    train_loss = n_.train_history['train_loss']
    test_accuracy = n_.train_history['test_accuracy']

    print(len(train_loss), '--', len(test_accuracy))

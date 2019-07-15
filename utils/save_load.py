import os
import pickle







def save_net(net, directory, file):
    net.save()


    
def save_object(params, directory, file_name):
    if not os.path.exists(directory):
	os.makedirs(directory)
    	with open(directory + file_name, 'wb') as f:
            pickle.dump(params, f)

def load_object(directory, file_name):
    with open(directory + file_name, 'rb') as f:
        return pickle.load(f)



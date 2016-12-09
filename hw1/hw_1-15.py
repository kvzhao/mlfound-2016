from mltools.PLA import PLA

data_path = 'data/hw1_15_train.dat'

pla = PLA()

pla.load_train_data(data_path)
pla.init_weight(mode='zero')
pla.train()
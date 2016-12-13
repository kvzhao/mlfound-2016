from mltools.PLA import PLA
import numpy as np
import matplotlib.pyplot as plt

print ('Prob 1-17')
data_path = 'data/hw1_15_train.dat'
pla = PLA()

pla.load_train_data(data_path)

mode = 'normal'
pla.init_weight(mode)
print ('Initialization method: ' + mode)

loop_mode = 'rand_cycle'

lr_sched = np.linspace(0.05, 1, 20)
num_of_experiments = 2000

meanw_vs_eta = []
updates_vs_eta = []

for lr in lr_sched:
	print ('eta = %f' % lr)
	updates_list = []
	mean_w = []
	for i in range(num_of_experiments):
		num_of_updates = pla.train(loop_mode, eta = lr)
		updates_list.append(num_of_updates)
		mean_w.append(np.mean(pla.W))
		pla.init_weight(mode)
	meanw_vs_eta.append(np.mean(mean_w))
	updates_vs_eta.append(np.mean(updates_list))

plt.plot(updates_vs_eta)
plt.xlabel('Learning Rate $\eta$')
plt.ylabel('Number of Updates')
plt.title('Learning Rate vs Number of Updates ('+ mode +' Initailization)')
plt.grid(True)
plt.savefig('results/pla-eta-vs-updates-'+mode+'.png')
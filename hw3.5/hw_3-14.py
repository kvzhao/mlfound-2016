from mltools.LinearRegressionModel import LinearRegressionModel
import numpy as np
import matplotlib.pyplot as plt

def main():
	pass

if __name__ == '__main__':
	train_data_path = 'data/hw4_train.dat'
	test_data_path = 'data/hw4_test.dat'

	LR = LinearRegressionModel()

	mode = 'zero'

        #lambdas = np.linspace(10**-10,10**3, 1000)
        log_lambda = range(-10,3)
        #log_lambda.reverse()
        lambdas = [10**i for i in log_lambda]

        Ein = []
        Eout = []
        for l in lambdas:
	    LR.load_train_data(train_data_path)
	    LR.init_weight(mode)
	    ein = LR.train_on(split_range=slice(0,200), lambda_val= l)
            Ein.append(ein)
	    LR.load_test_data(test_data_path)
	    eout = LR.Eout()
            Eout.append(eout)

        print ('minimum Ein = %f, minimum Eout = %f' % (np.min(Ein), np.min(Eout)))
        l_min_ein = lambdas[np.argmin(Ein)]
        print ('lambda with minimum Ein is l=%e , corresponding Eout=%f' % (l_min_ein, Eout[np.argmin(Ein)]))
        l_min_eout = lambdas[np.argmin(Eout)]
        print ('lambda with minimum Eout is l=%e , corresponding Ein=%f' % (l_min_eout, Ein[np.argmin(Eout)]))

        plt.semilogx(lambdas, Ein, '--o', linewidth=2, label='Ein')
        plt.semilogx(lambdas, Eout, '--o', linewidth=2, label='Eout')
        plt.legend(['Ein', 'Eout'], loc=2)
        plt.xlabel('$\lambda$')
        plt.ylabel('Errors')
        plt.title('Fig - 14')
        plt.grid(True)
        plt.savefig('results/figure_14.png')
        plt.show()
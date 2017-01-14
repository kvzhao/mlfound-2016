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
        log_lambda = range(-10, 3)
        lambdas = [10**i for i in log_lambda]

        Eout = []
        Ein = []
        Ecv = []
        for l in lambdas:
	    LR.load_train_data(train_data_path)
	    LR.init_weight(mode)
            ecv = LR.cross_validate_train(folds=5, lambda_val=l)
            Ecv.append(ecv)
	    LR.load_test_data(test_data_path)
	    eout = LR.Eout()
            Eout.append(eout)

        print ('minimum Ecv = %f minimum Eout = %f' % (np.min(Ecv), np.min(Eout)))
        l_min_ecv = lambdas[np.argmin(Ecv)]
        print ('lambda with minimum Ecv is l=%e , corresponding Eout=%f' % (l_min_ecv, Eout[np.argmin(Ecv)]))
        l_min_eout = lambdas[np.argmin(Eout)]
        print ('lambda with minimum Eout is l=%e , corresponding Ecv=%f' % (l_min_eout, Ecv[np.argmin(Eout)]))


        plt.semilogx(lambdas, Ecv, '--^', markersize=8,linewidth=2, label='Ecv')
        plt.semilogx(lambdas, Eout, '--s', markersize=8,linewidth=2, label='Eout')

        plt.legend(['Ecv', 'Eout'], loc=2)
        plt.xlabel('$\lambda$')
        plt.ylabel('Errors')
        plt.title('Fig - 19')
        plt.grid(True)
        plt.savefig('results/figure_19.png')
        plt.show()

        # Run optimal lambda

        opt_lambda = l_min_ecv
        LR.load_train_data(train_data_path)
	LR.init_weight(mode)
	opt_ein = LR.train(lambda_val= opt_lambda)
	opt_eout = LR.Eout()
        print('Run optimal lambda = %e on Dtrain, Ein=%f, Eout=%f' % (opt_lambda, opt_ein, opt_eout))
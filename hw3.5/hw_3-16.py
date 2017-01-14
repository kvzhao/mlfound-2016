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

        Ein = []
        Eval = []
        Eout = []
        for l in lambdas:
	    LR.load_train_data(train_data_path)
	    LR.init_weight(mode)
	    ein = LR.train_on(split_range=slice(0,120), lambda_val= l)
	    ev = LR.eval_on(split_range=slice(120,201))
            Ein.append(ein)
            Eval.append(ev)
	    LR.load_test_data(test_data_path)
	    eout = LR.Eout()
            Eout.append(eout)

        print ('minimum Ein = %f, minimum Eval = %f minimum Eout = %f' % (np.min(Ein), np.min(Eval), np.min(Eout)))
        l_min_ein = lambdas[np.argmin(Ein)]
        print ('lambda with minimum Ein is l=%e , corresponding Eval=%f, Eout=%f' % (l_min_ein, Eval[np.argmin(Ein)], Eout[np.argmin(Ein)]))
        l_min_eval = lambdas[np.argmin(Eval)]
        print ('lambda with minimum Eval is l=%e , corresponding Ein=%f, Eout=%f' % (l_min_eval, Ein[np.argmin(Eval)], Eout[np.argmin(Eval)]))
        l_min_eout = lambdas[np.argmin(Eout)]
        print ('lambda with minimum Eout is l=%e , corresponding Ein=%f, Eval=%f' % (l_min_eout, Ein[np.argmin(Eout)], Eval[np.argmin(Eout)]))

        plt.semilogx(lambdas, Ein, '--o', markersize=8, linewidth=2, label='Ein')
        plt.semilogx(lambdas, Eval, '--^', markersize=8,linewidth=2, label='Eval')
        plt.semilogx(lambdas, Eout, '--s', markersize=8,linewidth=2, label='Eout')

        #plt.plot(log_lambda, Ein, linewidth=2, label='Ein')
        #plt.plot(log_lambda, Eval, linewidth=2, label='Eval')
        #plt.plot(log_lambda, Eout, linewidth=2, label='Eout')

        plt.legend(['Etrain', 'Eval', 'Eout'], loc=2)
        plt.xlabel('$\lambda$')
        plt.ylabel('Errors')
        plt.title('Fig - 16')
        plt.grid(True)
        plt.savefig('results/figure_16.png')
        plt.show()

        # Run optimal lambda

        opt_lambda = l_min_eout
        LR.load_train_data(train_data_path)
	LR.init_weight(mode)
	opt_ein = LR.train(lambda_val= opt_lambda)
	opt_eout = LR.Eout()
        print('Run optimal lambda = %e on Dtrain, Ein=%f, Eout=%f' % (opt_lambda, opt_ein, opt_eout))
# !/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os
import sys
import pickle

sys.path.insert(0, '/home/arpit/Projects/Bayesian-neural-transfer-learning/preliminary/WineQualityDataset/preprocess/')
from preprocess import getdata

def convert_time(secs):
    if secs >= 60:
        mins = str(int(secs/60))
        secs = str(int(secs%60))
    else:
        secs = str(int(secs))
        mins = str(00)

    if len(mins) == 1:
        mins = '0'+mins

    if len(secs) == 1:
        secs = '0'+secs

    return [mins, secs]

# --------------------------------------------------------------------------

# An example of a class
class Network:
    def __init__(self, Topo, Train, Test, learn_rate, alpha):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed(int(time.time()))
        self.lrate = learn_rate
        self.alpha = alpha
        self.NumSamples = self.TrainData.shape[0]

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
        self.W3 = np.random.randn(self.Top[2], self.Top[3]) / np.sqrt(self.Top[2])
        self.B3 = np.random.randn(1, self.Top[3]) / np.sqrt(self.Top[2])  # bias second layer



        self.hidout1 = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.hidout2 = np.zeros((1,self.Top[2]))
        self.out = np.zeros((1, self.Top[3]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[3]
        return sqerror

    def sampleAD(self, actualout):
        error = np.subtract(self.out, actualout)
        moderror = np.sum(np.abs(error)) / self.Top[3]
        return moderror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout1 = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout1.dot(self.W2) - self.B2
        self.hidout2 = self.sigmoid(z2)  # output second hidden layer
        z3 = self.hidout2.dot(self.W3) - self.B3
        self.out = self.sigmoid(z3)

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta2 = out_delta.dot(self.W3.T) * (self.hidout2 * (1 - self.hidout2))
        hid_delta1 = hid_delta2.dot(self.W2.T) * (self.hidout1 * (1 - self.hidout1))

        self.W3 += (self.hidout2.T.dot(out_delta) * self.lrate)
        self.B3 += (-1 * self.lrate * out_delta)
        self.W2 += (self.hidout1.T.dot(hid_delta2) * self.lrate)
        self.B2 += (-1 * self.lrate * hid_delta2)
        self.W1 += (Input.T.dot(hid_delta1) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta1)

        # layer = 1  # hidden to output
        # for x in xrange(0, self.Top[layer]):
        #     for y in xrange(0, self.Top[layer + 1]):
        #         self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
        # for y in xrange(0, self.Top[layer + 1]):
        #     self.B2[y] += -1 * self.lrate * out_delta[y]
        #
        # layer = 0  # Input to Hidden
        # for x in xrange(0, self.Top[layer]):
        #     for y in xrange(0, self.Top[layer + 1]):
        #         self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
        # for y in xrange(0, self.Top[layer + 1]):
        #     self.B1[y] += -1 * self.lrate * hid_delta[y]

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]
        w_layer3size = self.Top[2] * self.Top[3]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))

        w_layer3 = w[w_layer2size:w_layer3size + w_layer2size]
        self.W3 = np.reshape(w_layer3, (self.Top[2], self.Top[3]))

        self.B1 = w[w_layer1size + w_layer2size + w_layer3size :w_layer1size + w_layer2size + w_layer3size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + w_layer3size + self.Top[1] :w_layer1size + w_layer2size + w_layer3size + self.Top[1] + self.Top[2]]
        self.B3 = w[w_layer1size + w_layer2size + w_layer3size + self.Top[1] + self.Top[2]:w_layer1size + w_layer2size + w_layer3size + self.Top[1] + self.Top[2] + self.Top[3]]


    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w3 = self.W3.ravel()
        w = np.concatenate([w1, w2, w3, self.B1, self.B2, self.B3])
        return w

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in xrange(0, depth):
            for i in xrange(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return  w_updated

    def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[3]))
        fx = np.zeros((size,self.Top[3]))

        for i in xrange(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[i] = self.out

        return fx

    def TestNetwork(self, phase, erTolerance):
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[3]))
        Output = np.zeros((1, self.Top[3]))
        if phase == 1:
            Data = self.TestData
        if phase == 0:
            Data = self.TrainData
        clasPerf = 0
        sse = 0
        testSize = Data.shape[0]
        self.W1 = self.BestW1
        self.W2 = self.BestW2  # load best knowledge
        self.W3 = self.BestW3
        self.B1 = self.BestB1
        self.B2 = self.BestB2  # load best knowledge
        self.B3 = self.BestB3


        for s in xrange(0, testSize):

            Input[:] = Data[s, 0:self.Top[0]]
            Desired[:] = Data[s, self.Top[0]:]

            self.ForwardPass(Input)
            sse = sse + self.sampleEr(Desired)

            if (np.isclose(self.out, Desired, atol=erTolerance).all()):
                clasPerf = clasPerf + 1

        return (sse / testSize, float(clasPerf) / testSize * 100)

    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestW3 = self.W3
        self.BestB1 = self.B1
        self.BestB2 = self.B2
        self.BestB3 = self.B3

    def plot_err(self, mse_lis, mad_lis, depth):

        ax = plt.subplot(111)
        plt.plot(range(depth), mse_lis, label="mse")
        plt.plot(range(depth), mad_lis, label="mad")

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.xlabel('Epoch')
        plt.ylabel('Mean Error')
        plt.title(' Mean Error')
        plt.savefig('error.png')

    def BP_GD(self, stocastic, vanilla, depth):  # BP with SGD (Stocastic BP)
        # self.momenRate = mRate
        # self.NumSamples = numSamples

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[3]))
        # Er = []#np.zeros((1, self.Max))
        epoch = 0
        bestmse = 100
        bestmad = 100
        bestTrain = 0

        mad_lis = []
        mse_lis = []

        train_acc = []
        test_acc = []
        train_sse = []
        test_sse = []

        start = time.time()
        # while  epoch < self.Max and bestTrain < self.minPerf :
        while epoch < depth:
            sse = 0
            sad = 0
            for s in xrange(0, self.NumSamples):

                if (stocastic):
                    pat = random.randint(0, self.NumSamples - 1)
                else:
                    pat = s

                Input[:] = self.TrainData[pat, :self.Top[0]]
                Desired[:] = self.TrainData[pat, self.Top[0]:]

                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)
                sse = sse + self.sampleEr(Desired)
                sad = sad + self.sampleAD(Desired)

            mse = np.sqrt(sse / self.NumSamples * self.Top[3])
            mad = sad / self.NumSamples * self.Top[3]

            if mse < bestmse:
                bestmse = mse
                self.saveKnowledge()
                (x, bestTrain) = self.TestNetwork(0, 0.2)

            if mad < bestmad:
                bestmad = mad

            mse_lis.append(bestmse)
            mad_lis.append(bestmad)

            if epoch%10 == 0:
                sse, acc = self.TestNetwork(phase=1, erTolerance=etol)
                test_acc.append(acc)
                test_sse.append(sse)

                sse, acc = self.TestNetwork(phase=0, erTolerance=etol_tr)
                train_acc.append(acc)
                train_sse.append(sse)

            elapsed = convert_time(time.time() - start)
            # Er = np.append(Er, mse)
            sys.stdout.write('\rEpoch: '+str(epoch+1)+"/"+ str(depth)+" mse: "+ str(mse) + " mad: " + str(mad) + " Time elapsed: "+str(elapsed[0])+":"+str(elapsed[1]))

            epoch = epoch + 1

        # self.plot_err(mse_lis, mad_lis, depth)

        # return (mse, mad, bestmse, bestTrain, epoch)
        return train_acc, test_acc

    def transfer_weights(self, source):
        self.W1 = source.W1
        self.W2 = source.W2
        self.W3 = source.W3
        self.B1 = source.B1
        self.B2 = source.B2
        self.B3 = source.B3

# --------------------------------------------------------------------------
class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print y_train.size
        print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.02;  # defines how much variation you need in changes to w
        step_eta = 0.01;
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)
        print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print likelihood

        naccept = 0
        print 'begin sampling using mcmc random walk'
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print    i, ' is accepted sample'
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro

                print  likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted'

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

        print naccept, ' num accepted'
        print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)

# --------------------------------------------------------------------------


def pickle_knowledge(obj, pickle_file):
    pickling_on = open(pickle_file, 'wb')
    pickle.dump(obj, pickling_on)
    pickling_on.close()

# --------------------------------------------------------------------------
if __name__ == '__main__':


    input = 11
    hidden1 = 90
    hidden2 = 90
    output = 4
    topo = [input, hidden1, hidden2, output]
    # print(traindata.shape, testdata.shape)
    lrate = 0.67
    etol_tr = 0.2
    etol = 0.6
    alpha = 0.1

    minepoch = 0
    maxepoch = 500
    traindata, testdata = getdata('WineQualityDataset/winequality-white.csv')

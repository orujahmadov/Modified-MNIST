import numpy
import sklearn
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import timeit

class myNN:

	def __init__(self, hiddenLayerSize, alpha, weights = 0, lam = 0, beta = 0, max_iter = 50, verbose = False):
		self.hiddenLayerSize = hiddenLayerSize
		self.alpha = alpha
		self.beta = beta
		self.lam = lam
		self.le = preprocessing.LabelEncoder()
		self.max_iter = max_iter
		self.verbose = verbose
		if type(weights) == list:
			self.weights = list(weights)
		else:
			self.weights = weights

	def train(self,x,y, batch_size = 0, cvx = [], cvy = []):
		if len(x) != len(y):
			raise NameError('Dimension Mismatch')

		self.inputLayerSize = len(x[0])
		self.numberOfObservations = len(x)
		self.outputLayerSize = len(numpy.unique(y))
		self.w_d = []
		self.epoch = 0
		self.rollOver = False
		self.dataIndex = 0
		self.batch_size = batch_size
		start = timeit.default_timer()
		if not self.le.get_params():
			self.le.fit(y)

		newY = self.le.transform(y)

		if len(numpy.asarray(cvy)) > 0:
			nCvy = self.le.transform(cvy)
		
		if type(self.hiddenLayerSize) == int:
			self.numberOfHiddenlayers = 1
		else:
			self.numberOfHiddenlayers = len(self.hiddenLayerSize)

		if self.weights == 0:
			self.weights = self.initWeights()
			print "Weights matrix initialized with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))
		else:
			print "Weights matrix already given with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))
			self.w_d = self.weights[:]
			for i in range(len(w_d)):
				w_d[i] = numpy.zeros(w_d[i].shape)

		if self.batch_size == 0:
			self.batch_size = self.numberOfObservations

		print "Alpha is set to {}".format(self.alpha)
		print "Batch size is set to {}".format(self.batch_size)

		w_r = self.roll(self.weights)
		dGrad = 1
		oldGrad = 0
		cAccuracy = 0
		cost = []
		accuracy = []
		cv_cost = []
		cv_accuracy = []

		while (self.epoch < self.max_iter): #(dGrad > 0.01) and (self.epoch < self.max_iter) and (cAccuracy < 0.9):

			while not self.rollOver:
				xTrain, yTrain = self.nextBatch(x,newY)
				grad = self.backPropogate(w_r,xTrain,yTrain)
				self.weights = self.UpdateWeights(xTrain,grad)
				w_r = self.roll(self.weights)

			dGrad = numpy.sum(numpy.abs(grad))
			cAccuracy = sklearn.metrics.accuracy_score(y,self.predict(x))
			cost.append(self.computeCost(w_r,x,newY))
			accuracy.append(cAccuracy)
			if cvx != []:
				cv_cost.append(self.computeCost(w_r,cvx,nCvy))
				cv_accuracy.append(sklearn.metrics.accuracy_score(cvy,self.predict(cvx)))
				suffix = 'acc = {0:.4f}, cost = {1:.4f}, cv_acc = {2:.4f}, cv_cost = {3:.4f}'.format(accuracy[self.epoch],cost[self.epoch],cv_accuracy[self.epoch],cv_cost[self.epoch])
			else:
				suffix = 'acc = {0:.4f}, cost = {1:.4f}'.format(accuracy[self.epoch],cost[self.epoch])
			self.epoch = self.epoch + 1
			self.printProgressBar(prefix = 'Training:', suffix = suffix)
			self.rollOver = False

		if self.verbose:
			filename = 'TrainResults_acc_{acc}_b_{b}_a_{a}_beta_{beta}_hidden_layer_{h}(2).csv'.format(acc = accuracy[self.epoch -1], b = self.batch_size, a =self.alpha, beta=self.beta,h=self.hiddenLayerSize)
			if cvx == []:
				numpy.savetxt(filename, numpy.c_[numpy.arange(len(accuracy)),accuracy,cost],delimiter=',',fmt='%3.4f')
			else:
				numpy.savetxt(filename, numpy.c_[numpy.arange(len(accuracy)),accuracy,cost,cv_accuracy,cv_cost],delimiter=',',fmt='%3.4f')

		stop = timeit.default_timer()
		print 'Training complete in {0:.4f} seconds in {1} epocs with a train accuracy of {2:.4f}'.format(stop-start, self.epoch,cAccuracy)
		return self.weights, cost, accuracy

	def initWeights(self):
		nInput = self.inputLayerSize
		nOutput = self.outputLayerSize
		nHidden = []
		w = []

		if type(self.hiddenLayerSize) == int:
			nHidden.append(self.hiddenLayerSize)
		else:
			nHidden = self.hiddenLayerSize

		size = self.numberOfHiddenlayers + 1
		for i in range(size):
			if i == 0:
				layerIn = nInput + 1
			else:
				layerIn = nHidden[i-1] + 1

			if (i + 1) == size:
				layerOut = nOutput
			else:
				layerOut = nHidden[i]

			epsilon = numpy.sqrt(6.0 / (layerIn + layerOut))
			w.append(2*epsilon*numpy.random.random((layerOut,layerIn)) - epsilon)
			self.w_d.append(numpy.zeros(w[i].shape))
		return w

	def nextBatch(self,x,y):
		if(self.dataIndex + self.batch_size > self.numberOfObservations):
			self.rollOver = True
			remaining = self.numberOfObservations - self.dataIndex
			x1 = []
			if self.dataIndex != self.numberOfObservations:
				x1 = x[self.dataIndex:self.numberOfObservations]
				y1 = y[self.dataIndex:self.numberOfObservations]
			idx0 = numpy.arange(self.numberOfObservations)
			x = x[numpy.random.shuffle(idx0)].reshape(-1,len(x[0]))
			y = y[numpy.random.shuffle(idx0)].reshape(len(y))
			self.dataIndex = self.batch_size - remaining
			if x1 != []:
				xReturn = numpy.concatenate((x1 , x[0:self.dataIndex]), axis = 0)
				yReturn = numpy.concatenate((y1 , y[0:self.dataIndex]), axis = 0)
			else:
				xReturn = x[0:self.dataIndex]
				yReturn = y[0:self.dataIndex]				
		else:
			xReturn = x[self.dataIndex:(self.dataIndex+self.batch_size)]
			yReturn = y[self.dataIndex:(self.dataIndex+self.batch_size)]
			self.dataIndex += self.batch_size
		return xReturn, yReturn

	def computeCost(self,w_r,x,y):
		layers = self.numberOfHiddenlayers + 1
		m = len(x)
		l = self.outputLayerSize 
		w = self.unRoll(w_r)
		cost = 0
		for i in range(m):
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
			# yI is the ideal y output
			yI = numpy.zeros(l)
			yI[y[i]] = 1.0

			cost = cost - numpy.sum((yI*numpy.log(cOutput)) + ((1-yI)*numpy.log(1-cOutput)))

		regCost = 0
		for i in range(layers):
		 	regCost = regCost + numpy.sum(numpy.sum(numpy.square(w[i]))) - numpy.sum(numpy.sum(numpy.square(w[i][:,0]))) 
		return (1.0/m)*cost + (self.lam/(2.0*m))*regCost

	def backPropogate(self,w_r,x,y):
		layers = self.numberOfHiddenlayers + 1
		m = len(x)
		l = self.outputLayerSize
		w = self.unRoll(w_r)
		grad = []
		# Go through every sample
		for i in range(m):
			cOutputs = []
			cInputs  = []
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
				cInputs.append(cInput)
				cOutputs.append(cOutput)
			yI = numpy.zeros(l)
			yI[y[i]] = 1.0

			# Compute Gradients going backwards 
			delta = []
			for j in reversed(range(layers)):
				sG = cOutputs[j]*(1-cOutputs[j])
				if (j+1) == layers:
					temp = sG * (yI - cOutputs[j])
				else:
					pW = numpy.dot(w[j+1].T,delta[layers-j-2])
					temp = sG*pW[1:]
				delta.append(temp)

			delta.reverse()

			# Update the gradient
			for j in range(len(w)):
				if i == 0:
					grad.append(numpy.array(numpy.matrix(delta[j]).T*numpy.matrix(cInputs[j])))
				else:
					grad[j] = grad[j] + numpy.array(numpy.matrix(delta[j]).T*numpy.matrix(cInputs[j]))

			#print "gradient yielded {}".format(grad)

		for i in range(len(grad)):
			regTerm = numpy.zeros(grad[i].shape)
			regTerm[:,1:] = grad[i][:,1:]
			grad[i] = (1.0/m)*grad[i] + (self.lam / m)*regTerm

		grad_r = self.roll(grad)

		return grad_r

	def UpdateWeights(self, x, grad_r):
		layers = self.numberOfHiddenlayers + 1
		m = len(x)
		w = self.weights
		w_d = self.w_d
		grad = self.unRoll(grad_r)
		for i in range(m):
			for j in range(layers):
				w_d[j] = grad[j]*self.alpha + w_d[j]*self.beta
				w[j] += w_d[j]
		return w

	def sigmoid(self,x):
		# Sigmoid function
		return 1. / (1 + numpy.exp(-x))

	def sigmoidGrad(self,x):
		#g = self.sigmoid(x)
		return numpy.dot(g, (1-g))

	def softmax(self, x):
		e_x = numpy.exp(x - numpy.max(x))
		return e_x / e_x.sum()

	def predict(self,x):
		if self.weights == 0:
			raise NameError('DidNotTrainYet')
		m = len(x)
		layers = self.numberOfHiddenlayers + 1
		w = self.weights
		y = []
		for i in range(m):
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
			# yI is the ideal y output
			yP = self.softmax(cOutput)
			y.append(self.le.inverse_transform((numpy.argmax(yP))))
		return y

	def roll(self,w):
		for i in range(self.numberOfHiddenlayers+1):
			temp = numpy.array(w[i]).ravel()
			if i == 0:
				rolledW = temp
			else:
				rolledW = numpy.concatenate((rolledW,temp))
		return rolledW

	def unRoll(self,w):
		unRolledW = []
		index = 0
		nInput = self.inputLayerSize
		nOutput = self.outputLayerSize
		size = self.numberOfHiddenlayers + 1
		nHidden = []

		if type(self.hiddenLayerSize) == int:
			nHidden.append(self.hiddenLayerSize)
		else:
			nHidden = self.hiddenLayerSize

		for i in range(size):
			if i == 0:
				layerIn = nInput + 1
			else:
				layerIn = nHidden[i-1] + 1

			if (i + 1) == size:
				layerOut = nOutput
			else:
				layerOut = nHidden[i]
			unRolledW.append(numpy.reshape(w[index:(index+layerIn*layerOut)],(layerOut,layerIn)))
			index = index+layerIn*layerOut
		return unRolledW
	
	def printProgressBar (self, prefix = '', suffix = '', decimals = 1, length = 25, fill = '#'):
		# Adapted from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
		iteration = self.epoch
		total = self.max_iter
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print '\r{p} |{b}| {pe}% {s}\r'.format(p = prefix, b = bar, pe = percent, s = suffix),
		# Print New Line on Complete
		if iteration == total: 
			print()

def debugMNIST():
	dataContent = sio.loadmat('ex4data1.mat')
	weights = sio.loadmat('ex4weights.mat')
	x = dataContent['X']
	y = dataContent['y'] % 10

	xTrain, xCV, yTrain, yCV = train_test_split(x, y, test_size=0.2, random_state=42)

	batches = [16]  	#[0,1,16,128]
	alphas = [1e-3] 	#[1e-3,1e-4,1e-5]

	for i in range(len(batches)):
		for j in range(len(alphas)):
			clf = myNN((25), alphas[j], lam = 0.5, beta = 0.5, verbose = True)
			w, c, a = clf.train(xTrain,yTrain,batch_size = batches[i], cvx = xCV, cvy = yCV)

	return clf

def main():
	x = numpy.loadtxt('train_x.csv',delimiter=',')
	y = numpy.loadtxt('train_y.csv', delimiter=',')

	xTrain, xCV, yTrain, yCV = train_test_split(x, y, test_size=0.20, random_state=42)

	batches = [128] #,16,128,1024,0]
	alphas = [1e-3] #[,1e-4,1e-2]
	hidden = [(200),(100),(150,75)]
	for i in range(len(batches)):
		for j in range(len(alphas)):
			for k in range(len(hidden)):
				clf = myNN(hidden[k], alphas[j], lam = 0.5, beta = 0.5, verbose = True)
				w, c, a = clf.train(xTrain,yTrain,batch_size = batches[i], cvx = xCV, cvy = yCV)
	return clf
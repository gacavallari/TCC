## Codigo exemplo para a Escola de Matematica Aplicada, minicurso Deep Learning
##
## Exemplo de autoencoder
##
## Moacir A. Ponti (ICMC/USP), Janeiro de 2018
## Referencia: Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask. Moacir A. Ponti, Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse
##

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

# 1) Definir arquitetura
Linp = 784
L1 = 128

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# Encoder (1 camada)
# matriz de pesos para cada camada
# random_normal para dados, truncated_normal para imagens
We = tf.Variable(tf.truncated_normal([Linp, L1], stddev=0.1)) # Linp features x L1 neuronios
be = tf.Variable(tf.truncated_normal([L1])) # bias de L1 feature maps

# Decoder (1 camada)
Wd = tf.Variable(tf.truncated_normal([L1, Linp], stddev=0.1)) # L1 x Linp neuronios
bd = tf.Variable(tf.truncated_normal([Linp]))     

# Modelo que ira gerar as predicoes
# Mutiplicacao matricial, soma de vetor, funcao de ativacao 
# TanH para dados entre -1 e 1, Sigmoidal para 0 a 1
# vetor de entrada
X1 = tf.reshape(X, [-1, Linp])


# representacao latente (code)
# obs:
C1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X1, We), be))

# formula para as predicoes
X_ = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(C1, Wd), bd))

# Define outras variaveis
batchSize = 100

# 2) Funcao de custo: informa quao longe estamos da solucao desejada
# Nao temos rotulos, entao devemos comparar com a entrada
# | X - X_ |^2 - erro medio quadratico, MSE
batch_mse   = tf.reduce_mean(tf.pow(X1 - X_, 2), 1)
mse   = tf.reduce_mean(tf.pow(X1 - X_, 2))
error = X1 - X_

# 3) Metodo de otimizacao e taxa de aprendizado
lrate = 0.0025
trainProcess = tf.train.RMSPropOptimizer(lrate).minimize(mse)

# Tudo pronto, agora podemos executar o treinamento
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iterations = 1000
# Dataset
DATA_DIR_MNIST = './data/mnist'
mnist = mnist_data.read_data_sets(DATA_DIR_MNIST, one_hot=True, reshape=False, validation_size=0)


###############################
#DATA_DIR_FASHION = './data/fashion'
#fashion = mnist_data.read_data_sets(DATA_DIR_FASHION, one_hot=True, reshape=False, validation_size=0)
###############################


# 4) Treinamento por iteracoes, cada iteracao realiza um
#    feed-forward na rede, computa custo, e realiza backpropagation
for i in range(iterations):
    # carrega batch de dados com respectivas classes
    batX, batY = mnist.train.next_batch(batchSize)
    # define dicionrio com pares: (exemplo,rotulo)
    trainData = {X: batX}
    # executa uma iteracao com o batch carregado
    sess.run(trainProcess, feed_dict=trainData)

    # computa acuracia no conjunto de treinamento e funcao de custo
    # (a cada 5 iteracoes)
    if (i%5 == 0):
        loss = sess.run(mse, feed_dict=trainData)
        print(str(i) + " Loss ="+str(loss))


# 5) Valida o modelo nos dados de teste (importante!)

testData = {X: mnist.test.images}
lossTest = sess.run(mse, feed_dict=testData)

testOriginal= mnist.test.images
testDecoded = sess.run(X_,feed_dict=testData)

print("\nTest Loss="+str(lossTest))
#print(testDecoded[1, ...])
#print(testOriginal[1, ...])


# 6) Extrai caracteristicas (codigos) e faz experimento com classificador SVM Linear (SVC)
# 	 com Cross Validation (10-fold)

#extrai labels do cojunto de teste no formato que a funcao cross_val_score necessita (one_hot=False)
mnist_ = mnist_data.read_data_sets(DATA_DIR_MNIST, one_hot=False, reshape=False, validation_size=0)
labels = mnist_.test.labels 
print ('\ndimensao da matriz de labels: '+ str(labels.shape))

#extrai codigos
codigos = sess.run(C1, feed_dict=testData)
print ('\ndimensao da matriz de codigos: '+ str(codigos.shape))


# estimate the accuracy of a linear kernel support vector machine, 
# by splitting the data, fitting a model and computing the score 10 consecutive times
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, codigos, labels, cv=10)
print ('\nscores: ' + str(scores))
print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



'''
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
	# imagem original
	ax = plt.subplot(2, n, i+1)
	#orig = np.reshape(testOriginal[i], (28,28))
	plt.imshow(testOriginal[i].reshape(28,28))
	#print(orig)
	#plt.imshow(orig)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i+1+n)
	#recon = np.reshape(testDecoded[i], (28,28))
	#print(recon)
	plt.imshow(testDecoded[i, ...].reshape(28,28))
	#plt.imshow(recon)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

lossTrain = "%.5f" % loss
lossTestOk = "%.5f" % lossTest
#plt.savefig('fashion_' + str(L1) + '_' + str(lossTrain) + '_' + str(lossTestOk) + '.png')
plt.savefig('fashion_mnist_' + str(L1) + '_' + str(lossTestOk) + '.png')
plt.close()
'''
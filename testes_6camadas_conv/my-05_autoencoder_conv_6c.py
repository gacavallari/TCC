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
L1 = 4
L2 = 392
L3 = 128

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# Encoder 1 convolucional 
We_1 = tf.Variable(tf.truncated_normal([3,3,1, L1], stddev=0.1)) # filtros 3x3x1, L1 neuronios/kernels 
be_1 = tf.Variable(tf.truncated_normal([L1]))      

#Encoder 2
We_2 = tf.Variable(tf.truncated_normal([28*28*L1, L2], stddev=0.1)) 
be_2 = tf.Variable(tf.truncated_normal([L2]))

#Encoder 3
We_3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1)) 
be_3 = tf.Variable(tf.truncated_normal([L3]))

#Decoder 1 
Wd_1 = tf.Variable(tf.truncated_normal([L3, L2], stddev=0.1))
#d_1 = tf.transpose(We_3)  
bd_1 = tf.Variable(tf.truncated_normal([L2]))

#Decoder 2
Wd_2 = tf.Variable(tf.truncated_normal([L2, 28*28*L1], stddev=0.1))
#d_1 = tf.transpose(We_2)  
bd_2 = tf.Variable(tf.truncated_normal([28*28*L1]))

#Decoder 3
Wd_3 = tf.Variable(tf.truncated_normal([3,3,1, L1], stddev=0.1)) 
#Wd_3 = tf.Variable(tf.truncated_normal([28*28*L1, 784], stddev=0.1)) 
#bd_3 = tf.Variable(tf.truncated_normal([784]))
bd_3 = tf.Variable(tf.truncated_normal([1]))


# vetor de entrada (agora como imagem)
X1 = tf.reshape(X, [-1, 28, 28, 1])
print("dimensao de X1: " + str(X1.shape))

C1 = tf.nn.conv2d(X1,We_1, strides=[1,1,1,1], padding="SAME") #"SAME"
C1 = tf.nn.relu(tf.nn.bias_add(C1, be_1))
C1 = tf.contrib.layers.flatten(C1)
print("dimensao de C1: " + str(C1.shape))

C2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(C1, We_2), be_2))
print("dimensao de C2: " + str(C2.shape))

C3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(C2, We_3), be_3))
print("dimensao de C3: " + str(C3.shape))

D1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(C3, Wd_1), bd_1))
print("dimensao de D1: " + str(D1.shape))

D2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(D1, Wd_2), bd_2))
print("dimensao de D2: " + str(D2.shape))

D3 = tf.reshape(D2, [-1, 28, 28, L1])
print("dimensao de D3: " + str(D3.shape))

# formula para as predicoes
X_ = tf.nn.conv2d_transpose(D3,Wd_3, output_shape=([tf.shape(X)[0], 28, 28, 1]),strides=[1,1,1,1], padding="SAME")
#X_ = tf.nn.conv2d_backprop_input(D3,Wd_3,strides=[1,1,1,1], padding="SAME")
print("dimensao de X_: " + str(X_.shape))
x_ = tf.nn.relu(tf.nn.bias_add(X_, bd_3))	
#X_ = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(D2, Wd_3), bd_3))
#X_ = tf.reshape(X_, [-1, 28, 28, 1])
print("dimensao de X_: " + str(X_.shape))

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

iterations = 10000
#loss = 100
#count_iterations = 0

# Dataset
#DATA_DIR = './data/mnist'
#mnist = mnist_data.read_data_sets(DATA_DIR, one_hot=True, reshape=False, validation_size=0)


###############################
DATA_DIR = './data/fashion'
mnist = mnist_data.read_data_sets(DATA_DIR, one_hot=True, reshape=False, validation_size=0)
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
mnist_ = mnist_data.read_data_sets(DATA_DIR, one_hot=False, reshape=False, validation_size=0)
labels = mnist_.test.labels 
print ('\ndimensao da matriz de labels: '+ str(labels.shape))

#extrai codigos
codigos = sess.run(C3, feed_dict=testData)
print ('\ndimensao da matriz de codigos: '+ str(codigos.shape))


# estimate the accuracy of a linear kernel support vector machine, 
# by splitting the data, fitting a model and computing the score 10 consecutive times
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, codigos, labels, cv=10)
print ('\nscores: ' + str(scores))
print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




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
plt.savefig('fashion_' + str(L3) + '_' + str(lossTestOk) + '.png')
plt.close()

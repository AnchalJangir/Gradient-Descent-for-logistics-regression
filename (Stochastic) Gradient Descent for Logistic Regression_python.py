import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)
#reading the data
X,y=datasets.load_svmlight_file(r"C:\Users\ancha\OneDrive\Documents\IIT Mandi 3rd Sem\P.P\a9a.txt")
X=X.toarray()

#adding a column of 1's in X
col_1=np.ones((X.shape[0],1))
X=np.append(X, col_1, axis=1)
y=y.reshape(-1,1)
print(X.shape)

#randomly assining dataset of 1000 points
idx=np.random.permutation(X.shape[0])[0:1000]
X=X[idx]
y=y[idx]
print(X.shape)

#Normalizing the data
mu=np.mean(X,axis=0)
sigma=np.std(X,axis=0)
X=(X-mu)/(sigma+1e-6)

#defining sigmoid funtion
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def J_fun(y,X,w):
   J=1+np.exp(-(y * (X@w)))
   return J

#Defining loss funtion 
def loss_fun(J):
    loss_function = np.mean(np.log(J))
    return loss_function

#Calculating gradient of the loss function w.r.t w 
def grad_fun_w(y,X,J):
    # Gradient of loss w.r.t weights.
    g = (1.0/X.shape[0])*(X.T @ (-((J-1)/J)*y))
    return g

#Applying gradient descent algorithm to find the optimal solution w,b in order to minimize loss function  
def grad_dec(X, y, N_iter, eta):
    # Empty list to store losses.
    loss_list= []
    # Empty list to store accuracy
    acc_list= []
    #Generating weight w randomly
    w=np.random.randn(X.shape[1],1)
    # Training loop.
    for epoch in range(N_iter+1):
            #predicting y 
            y_p = sigmoid(X@w)
            #calculating Accuracy 
            acc=np.mean((y_p>0.5) == (y>0))
            
            J=J_fun(y,X,w)
            #Calculating gradient of the loss function w.r.t w 
            g = grad_fun_w(y, X,J)
                        
            #Updating the parameters 
            w -= (eta*g + 0.1*w)
                  
            # Calculating loss and appending it in the list.
            loss=loss_fun(J)

            print("Epoch-{}, loss={:4f}, acc={:2f}%".format(epoch,loss,acc*100))
            loss_list.append(loss)
            acc_list.append(acc*100)

    #Plotting loss function and accuracy w.r.t no of iteration
    l=np.array(loss_list)
    a=np.array(acc_list)
    time=np.arange(0,N_iter+1,1)
    
    plt.plot(time,l)
    #plt.plot(time,np.log10(l - min(l) + 1e-6)/(1 + min(l)))
    plt.title('Loss vs time ')
    plt.xlabel('time')
    plt.ylabel('Loss function')
    plt.show()
    plt.plot(time,a)
    plt.title("Accuracy w.r.t time ")
    plt.xlabel('time')
    plt.ylabel('Accuracy')
    plt.show()
grad_dec(X, y, 100, 1e-3 )

#Applying Stochastic Gradient Descent algorithm to find the optimal solution w,b in order to minimize loss function  
def sgd(X,y,batch_size,N_iter,eta):
    #Empty list to store losses.
    loss_list= []
    #Empty list to store accuracy
    acc_list= []
    # Generating weight w randomly
    w=np.random.randn(X.shape[1],1)
    
    for epoch in range(N_iter+1):
       for i in range(((X.shape[0]-1)//batch_size)+ 1):
            
            #Defining  range of x and y batch
            x_batch = X[i*batch_size: i*batch_size+ batch_size]
            y_batch = y[i*batch_size: i*batch_size+ batch_size]
            
            # Predicitng y
            y_p = sigmoid(x_batch@w)
            
            J=J_fun(y_batch,x_batch,w)
            #Getting gradient of the loss function w.r.t w 
            g = grad_fun_w(y_batch, x_batch,J)
            
            # Updating the parameters
            w -= (eta*g + 0.1*w)
       # Calculating accuracy
       acc=np.mean((y_p>0.5) == (y_batch>0))    
       # Calculating loss and appending it in the list.
       loss=loss_fun(J)

       print("Epoch-{}, loss={:4f}, acc={:2f}%".format(epoch,loss,acc*100))
       loss_list.append(loss)
       acc_list.append(acc*100)

    #Plotting loss function and accuracy w.r.t no of iteration
    l=np.array(loss_list)
    a=np.array(acc_list)
    
    time=np.arange(0,N_iter+1,1)
    plt.plot(time,l)
    plt.title('Loss vs time')
    #plt.plot(time,np.log10(l - min(l) + 1e-6)/(1 + min(l)))
    plt.title('Loss vs time')
    plt.xlabel('time')
    plt.ylabel('Loss function')
    plt.show()
    plt.plot(time,a)
    plt.title("Accuracy w.r.t time")
    plt.xlabel('time')
    plt.ylabel('Accuracy')
    plt.show()
sgd(X, y, 100,200, 1e-3)

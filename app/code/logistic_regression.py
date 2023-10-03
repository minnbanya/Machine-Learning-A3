from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt 
import math
import time

class LogisticRegression:
    
    def __init__(self,regularization, k, n, method, alpha = 0.001,theta_init='zeros', max_iter=5000):
        self.regularization = regularization
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta_init = theta_init
        self.method = method
        # self.pre = []  # Initialize as empty lists
        # self.re = []
        # self.f1 = []
        # self.ratio = []
        # self.support = []
    
    def fit(self, X, Y):
        # self.W = np.random.rand(self.n, self.k)
        if self.theta_init == 'zeros':
            self.W = np.zeros((self.n, self.k))
        elif self.theta_init == 'xavier':
            m = X_train.shape[0]
            # calculate the range for the weights
            lower , upper = -(1.0 / math.sqrt(m)), (1.0 / math.sqrt(m))
            # randomly pick weights within this range
            # generate random numbers
            numbers = np.random.rand((self.n, self.k))
            self.W = lower + numbers * (upper - lower)
            
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            # batch_size = 32
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m + self.regularization(self.W)
        error = h - Y
        grad = self.softmax_grad(X, error)  + self.regularization.derivation(self.W)
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    # Accuracy function of the model
    def accuracy(self, X, Y):
        # accuracy = correct pred / total pred
        yhat = self.predict(X)
        return np.sum(yhat == Y)/ yhat.shape[0]
    
    def confusion_matrix(self, ytrue, ypred):
        # convert arrays to lists to allow easy data manipulation
        y_p = ypred.tolist()
        y_a = ytrue.tolist()

        # initialize empty confusion matrix
        cm = [[0 for x in range(self.k)] for y in range(self.k)]

        # ensure numnber of prediction and actual values are the same
        assert len(y_p) == len(y_a)

        # fill the confusion matrix
        for i in range (len(y_p)):
            for j in range (self.k):
                for l in range(self.k):
                    if y_a[i] == j and y_p[i] == l:
                        cm[j][l] += 1
        
        # true positive list (when cm[i][i])
        self.tp = [0 for x in range(self.k)]
        for i in range (self.k):
            self.tp[i] = cm[i][i]

        # true negative list is no required for precision and recall
        # false negative list
        self.fn = [0 for x in range (self.k)]
        for i in range (self.k):
            for j in range (self.k):
                if j != i:
                    self.fn[i] += cm[i][j]
        
        # false positive list
        self.fp = [0 for x in range (self.k)]
        for i in range (self.k):
            for j in range (self.k):
                if j != i:
                    self.fp[i] += cm[j][i]
    
    def precision(self):
        self.pre = [0 for x in range (self.k)]
        for i in range (self.k):
            if (self.tp[i] + self.fp[i]) == 0:
                self.pre[i] = 0
            else:
                self.pre[i] = self.tp[i] / (self.tp[i] + self.fp[i])
        return self.pre
    
    def recall(self):
        self.re = [0 for x in range (self.k)]
        for i in range (self.k):
            if (self.tp[i] + self.fp[i]) == 0:
                self.re[i] = 0
            else:
                self.re[i] = self.tp[i] / (self.tp[i] + self.fn[i])
        return self.re
    
    def f1_score(self):
        self.f1 = [0 for x in range (self.k)]
        for i in range (self.k):
            if self.pre[i] == 0 and self.re[i] == 0:
                self.f1[i] = 0
            else:
                self.f1[i] = (2 * self.pre[i] * self.re[i]) / (self.pre[i] + self.re[i])
        return self.f1
    
    def macro_precision(self):
        return sum(self.pre) / len(self.pre)
    
    def macro_recall(self):
        return sum(self.re) / len(self.re)
    
    def macro_f1(self):
        return sum(self.f1) / len(self.f1)
    
    def weighted_precision(self):
        denominator = 0
        for i in range (self.k):
            denominator += self.support[i]*self.pre[i]
        return denominator / self.Y_num
    
    def weighted_recall(self):
        denominator = 0
        for i in range (self.k):
            denominator += self.support[i]*self.re[i]
        return denominator / self.Y_num
    
    def weighted_f1(self):
        denominator = 0
        for i in range (self.k):
            denominator += self.support[i]*self.f1[i]
        return denominator / self.Y_num
    
    def classification_report_scratch(self, X, Y):
        # Method to show classification report similar to sklearn
                    
        # Call confusion matrix method to set TP,TN,FP,FN
        self.confusion_matrix(Y, self.predict(X))
        print(self.tp, self.fp, self.fn)

        # Calculate ratio of y for weighted calculations
        self.ratio = Y.value_counts(normalize=True, sort=False)

        # Support for classification report
        self.support = Y.value_counts(normalize=False, sort=False)

        # Total number of samples
        self.Y_num = Y.shape[0]

        # Call all methods to store values
        self.precision()
        self.recall()
        self.f1_score()
        macro_precision = self.macro_precision()  # Call these methods
        macro_recall = self.macro_recall()        # to calculate their values
        macro_f1 = self.macro_f1()                # before using them
    

        print("=========Classification report scratch=======")
        print("\t\tprecision \trecall \tf1 \tsupport\n")
        for i in range(self.k):
            print(f"{i} \t\t{self.pre[i]:.2f} \t{self.re[i]:.2f} \t{self.f1[i]:.2f} \t{self.support[i]}")
        print(f"\naccuracy \t\t\t{self.accuracy(X, Y):.2f} \t{Y.shape[0]}")
        print(f"macro avg \t{macro_precision:.2f} \t{macro_recall:.2f} \t{macro_f1:.2f} \t{Y.shape[0]}")
        print(f"weighted avg \t{self.weighted_precision():.2f} \t{self.weighted_recall():.2f} \t{self.weighted_f1():.2f} \t{Y.shape[0]}")

    def _coef(self):
        return self.W[1:]
    
    def feature_importance(self):
        feature_names = ['max_power',
                            'year',
                            'fuel',
                            'Ashok',
                            'Audi',
                            'BMW',
                            'Chevrolet',
                            'Daewoo',
                            'Datsun',
                            'Fiat',
                            'Force',
                            'Ford',
                            'Honda',
                            'Hyundai',
                            'Isuzu',
                            'Jaguar',
                            'Jeep',
                            'Kia',
                            'Land',
                            'Lexus',
                            'MG',
                            'Mahindra',
                            'Maruti',
                            'Mercedes-Benz',
                            'Mitsubishi',
                            'Nissan',
                            'Opel',
                            'Peugeot',
                            'Renault',
                            'Skoda',
                            'Tata',
                            'Toyota',
                            'Volkswagen',
                            'Volvo']
        importance_values = np.mean(np.abs(self._coef()[0:34]), axis=1)

        # Create a bar chart for feature importance
        plt.figure(figsize=(8, 6))
        plt.barh(feature_names, importance_values, color='blue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Graph')
        plt.xlim([0,np.max(self._coef())])  # Set the x-axis limits
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        plt.show()

class NormalPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return 0 # returning zero because normal just means using linear regression with no regularization
        
    def derivation(self, theta):
        return 0 # returning zero because normal just means using linear regression with no regularization

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Normal(LogisticRegression):
    
    def __init__(self, k, n, method, alpha=0.001,theta_init='zeros', l=0.1):
        self.regularization = NormalPenalty(l)
        super().__init__(self.regularization, k, n, method, alpha,theta_init='zeros')

class Lasso(LogisticRegression):
    
    def __init__(self, k, n, method, alpha=0.001,theta_init='zeros', l=0.1):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, k, n, method, alpha,theta_init='zeros')
        
class Ridge(LogisticRegression):
    
    def __init__(self, k, n, method, alpha=0.001,theta_init='zeros', l=0.1):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, k, n, method, alpha,theta_init='zeros')
        
class ElasticNet(LogisticRegression):
    
    def __init__(self, k, n, method, alpha=0.001,theta_init='zeros', l=0.1, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, k, n, method, alpha,theta_init='zeros')

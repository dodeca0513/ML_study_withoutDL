import numpy
from matplotlib import pyplot as plt

import matplotlib

from sklearn.datasets import make_blobs

#데이터 만들기
n_dim = 2
x_train, y_train = make_blobs(n_samples=100, n_features=n_dim, centers=[[1,1],[-1,-1]], shuffle=True)
x_test, y_test = make_blobs(n_samples=100, n_features=n_dim, centers=[[1,1],[-1,-1]], shuffle=True)
print(x_train)
print(y_train)


#모델 정의

def sigmoid(a):
    return 1./(1.+numpy.exp(-a))

def logreg(x,w,pre=False):#추론
    x=x.reshape([1,-1]) if len(x.shape)<2 else x
    y=numpy.sum(x*w[None,:-1],axis=1)+ w[-1]# x0*w0+x1*w1+b (※ w[-1]=bias)
    if pre:
        return y
    return sigmoid(y)

def logreg_dist(y,x,w,avg=False):#손실함수 계산
    y_=logreg(x,w)
    d=-(y*numpy.log(y_)+(1.-y)*numpy.log(1-y_))#손실값
    if not avg:
        return d
    return numpy.mean(d)

def logreg_rule(y,x,w):#학습방법
    y_=logreg(x,w)
    dw=numpy.zeros(w.shape)
    dw[:-1]=numpy.mean((y_-y)[:,None]*x,axis=0)#미분값
    dw[-1]=numpy.mean(y_-y)
    return dw
#시각화
def visual_data(x,y=None,c='r'):
    if y is None:
        y=[None]*len(x)
    for x_,y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0],x_[1],'ko')
        else:
            plt.plot(x_[0],x_[1],c+'o' if y_<0.5 else c+'+')
    plt.grid('on')
def visual_hyperplane(w,typ='k--'):

    m0,m1=-5,5

    intercept0=-(w[0]*m0+w[-1])/w[1]
    intercept1=-(w[0]*m1+w[-1])/w[1]

    plt.plot([m0,m1],[intercept0,intercept1],typ)
#학습
w0=numpy.random.randn(n_dim+1)
w0[-1]=0
w=numpy.copy(w0)
visual_hyperplane(w0,"r--")


n_iter=1000
eta=.1 # = lr
old_cost=numpy.inf
for ni in range(n_iter):
    #pred_y=logreg(x_train,w)
    w-=eta*logreg_rule(y_train,x_train,w)
    cost=logreg_dist(y_train,x_train,w,avg=True)
    if numpy.mod(ni,100)==0:
        print('Logistic regression cost {:.6f} after iteration {}'.format(cost,ni))
        visual_hyperplane(w)
    if cost<1e-16 or cost>=old_cost:
        print("Converged")
        break
    old_cost=cost

visual_data(x_train,y_train)
visual_hyperplane(w,"b-")
plt.show()

#테스트

pre=logreg(x_test,w)
print("test data distance =",logreg_dist(y_test,x_test,w,True))
visual_data(x_test,pre,"g")
visual_hyperplane(w,"b-")
plt.show()
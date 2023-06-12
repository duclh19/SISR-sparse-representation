import numpy as np 
from scipy import sparse

def sparse_solution(lmbd, A, b, maxiter):
    
    eps = 1e-9
    x = np.zeros((A.shape[0], 1))
    
    grad = np.dot(A,x)+b
    ma = np.max(np.abs(grad))
    mi = np.argmax(np.abs(grad))
    cnt_1 = 0
    cnt_2 = 0
    while True:
        if grad[mi]>lmbd+eps:
            x[mi] = (lmbd-grad[mi])/A[mi,mi]
        elif grad[mi]<-lmbd-eps:
            x[mi] = (-lmbd-grad[mi])/A[mi,mi]
        else:
            if np.all(x == 0):
                break
        while True:
            a = np.where(x != 0)[0] # active set
            Aa = A[np.repeat(a,len(a)),np.tile(a,len(a))].reshape(len(a),len(a))
            ba = b[a]
            xa = x[a]
            
            # new b based on unchanged sign
            vect = -lmbd*np.sign(xa)-ba
            if Aa.shape[0]==1:
                x_new = vect/Aa
            else:
                x_new = np.dot(np.linalg.inv(Aa),vect)
            idx = np.where(x_new != 0)[0]
            o_new = np.dot((vect[idx] / 2 + ba[idx]).T, x_new[idx]) + lmbd * np.sum(np.abs(x_new[idx]))
            
            # cost based on changing sign
            s = np.where(xa*x_new <= 0)[0]
            
            cnt_2 += 1
            if np.all(s == 0) or cnt_2>maxiter:
                x[a] = x_new
                cnt_2 = 0
                break
            x_min = x_new
            o_min = o_new
            d = x_new - xa
            
            t = d/xa
            for zd in s.T:
                x_s = xa - d / t[zd]
                x_s[zd] = 0
                idx = np.where(x_s != 0)[0]
                o_s = np.dot((np.dot(Aa[idx, idx], x_s[idx]) / 2 + ba[idx]).T, x_s[idx]) + lmbd * np.sum(np.abs(x_s[idx]))
                if o_s < o_min:
                    x_min = x_s
                    o_min = o_s
            
            x[a] = x_min
            
        grad = np.dot(A, x) + b
            
        temp = np.abs(grad)*(x == 0)
        ma = np.max(np.abs(temp))
        mi = np.argmax(np.abs(temp))
        
        cnt_1 += 1
        if ma<=lmbd+eps or cnt_1>maxiter:
            break
    return x



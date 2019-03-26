
# FM-FTRL

This is an implementation of the FM update with FTRL-Proximal algorithm in C with python bindings. FTRL-Proximal is an algorithm for online learning which is quite successful in solving sparse problems. The implementation is based on the algorithm from the ["Ad Click Prediction: a View from the Trenches"](https://research.google.com/pubs/pub41159.html) paper.

Some of the features:

* The python code can operate directly on scipy CSR matrices

## Pre-requisites 

Dependensies:

* It needs: `numpy`, `scipy` and open mp
* If you use anaconda, it already has  `numpy`, `scipy`
* to install `GOMP_4.0` for anaconda, use `conda install libgcc`


## Building

    cmake . && make
    mv libfm.so ftrl/

If you don't have `cmake`, it's easy to install:

    mkdir cmake && cd cmake
    wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.sh
    bash cmake-3.10.0-Linux-x86_64.sh --skip-license
    export CMAKE_HOME=`pwd`
    export PATH=$PATH:$CMAKE_HOME/bin


## Example

    import numpy as np
    import scipy.sparse as sp

    from sklearn.metrics import roc_auc_score

    import ftrl

    X = [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],   
    ]

    X = sp.csr_matrix(X, dtype='float32')
    y = np.array([1, 1, 1, 0, 0], dtype='float32')
    
    model = ftrl.FtrlProximal(d0=0, d1=0, d2=1, alpha=0.1, beta=0.1, l1=0.1, l2=0.1, model_type='classification')

    # make 20 passes over the data
    for i in range(20):
        log_loss = model.fit(X, y)
        print "log_loss::", log_loss
        y_pred = model.predict(X)
        print y_pred
        auc = roc_auc_score(y, y_pred)
        print("auc::{}".format(auc))


We can also use it to solve the regression problem:

    from sklearn.metrics import mean_squared_error

    y = np.array([1, 2, 3, 4, 5], dtype='float32')

    model = ftrl.FtrlProximal(d0=0, d1=0, d2=1, alpha=0.1, beta=0.1, l1=0.1, l2=0.1, model_type='regression')

    # make 20 passes over the data
    for i in range(20):
        mse = model.fit(X, y)
        print model.predict(X)
        print("mse::{}".format(mse))

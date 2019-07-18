import numpy as np

def compute_W(XWithBias, Y):
    Xt = XWithBias.T
    XtX = np.matmul(Xt,XWithBias)
    XtXInv = np.linalg.inv(XtX)
    PseudoInv = np.matmul(XtXInv,Xt)
    return np.matmul(PseudoInv,Y)

def add_input_bias(X):
    XWithBias = np.ones((X.shape[0], X.shape[1] + 1))
    XWithBias[:, 1:] = X
    return XWithBias

def create_linear_model(X, Y):
    XWithBias = add_input_bias(X)
    return compute_W(XWithBias, Y)

def predict(inputs, W):
    return np.matmul(inputs, W)

X = np.array(
    [
        [0],
        [1]
    ]
)

Y = np.array(
    [
        [2],
        [5]
    ]
)
W = create_linear_model(X, Y)

print(f"weights : {W}")

print(f"expected : {Y}")

print(f"results : {predict(add_input_bias(X), W)}")


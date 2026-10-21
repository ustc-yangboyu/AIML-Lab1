import numpy as np
import sklearn.linear_model

class LinearRegression:
    def __init__(self, LearningRate = 0.01, MaxIters = 1000, Lambda1 = 0.01, Lambda2 = 0.01):
        self.LearningRate = LearningRate
        self.MaxIters = MaxIters
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.Weights = None
        self.Bias = None
        self.Mean = None
        self.Std = None

    def LossFunc(self, Y_pred, Y):
        Mse_Loss = np.mean((Y_pred - Y) ** 2)
        return Mse_Loss

    def fit(self, X, Y, Log = [], Stopfunc = None, use_prox_l1 = False):
        n_samples, n_features = X.shape
        self.Weights = np.random.randn(n_features)
        self.Bias = 0
        self.Mean = np.mean(X, axis = 0)
        self.Std = np.std(X, axis = 0)

        X = (X - self.Mean) / self.Std

        if 'loss' in Log:
            Losses = []

        for i in range(self.MaxIters):
            Y_pred = X @ self.Weights + self.Bias

            Mse_Loss = np.mean((Y_pred - Y) ** 2)
            Loss = Mse_Loss + self.Lambda1 * np.sum(np.abs(self.Weights)) + self.Lambda2 * np.sum(self.Weights ** 2)

            dW = (2 / n_samples) * (X.T @ (Y_pred - Y)) + self.Lambda1 * np.sign(self.Weights) + 2 * self.Lambda2 * self.Weights
            dB = np.mean(Y_pred - Y)

            self.Weights -= self.LearningRate * dW
            self.Bias -= self.LearningRate * dB

            if use_prox_l1:
                self.Weights[np.abs(self.Weights) < self.LearningRate * self.Lambda1] = 0

            if 'loss' in Log:
                Losses.append(Loss)

            if Stopfunc is not None:
                if Stopfunc(Loss):
                    break

        if 'loss' in Log:
            return Losses
        else:
            return None

    def predict(self, X):
        X = (X - self.Mean) / self.Std
        if type(X) == np.ndarray:
            return X @ self.Weights + self.Bias
        elif type(X) == list:
            return np.array(X) @ self.Weights + self.Bias
        elif type(X) == int or type(X) == float:
            return X * self.Weights + self.Bias

class LogisticRegression:
    def __init__(self, LearningRate = 0.01, MaxIters = 1000, Lambda1 = 0.01, Lambda2 = 0.01):
        self.LearningRate = LearningRate
        self.MaxIters = MaxIters
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.Weights = None
        self.Bias = None
        self.Mean = None
        self.Std = None

    def Sigmoid(self, z):
        return np.exp(-np.logaddexp(0.0, -z))

    def LossFunc(self, Y_pred, Y):
        Bce_Loss  = np.mean(Y * np.logaddexp(0.0, -Y_pred) + (1 - Y) * np.logaddexp(0.0, Y_pred))
        return Bce_Loss

    def fit(self, X, Y, Log = [], Stopfunc = None, use_prox_l1 = False):
        n_samples, n_features = X.shape
        self.Weights = np.random.randn(n_features)
        self.Bias = 0
        self.Mean = np.mean(X, axis = 0)
        self.Std = np.std(X, axis = 0)

        X = (X - self.Mean) / self.Std

        if 'loss' in Log:
            Losses = []

        for _ in range(self.MaxIters):
            linear_out = X @ self.Weights + self.Bias
            Y_pred = self.Sigmoid(linear_out)

            Bce_Loss  = np.mean(Y * np.logaddexp(0.0, -linear_out) + (1 - Y) * np.logaddexp(0.0, linear_out))
            Loss = Bce_Loss + self.Lambda1 * np.sum(np.abs(self.Weights)) + self.Lambda2 * np.sum(self.Weights ** 2)

            dW = (1 / n_samples) * (X.T @ (Y_pred - Y)) + self.Lambda1 * np.sign(self.Weights) + 2 * self.Lambda2 * self.Weights
            dB = np.mean(Y_pred - Y)

            self.Weights -= self.LearningRate * dW
            self.Bias -= self.LearningRate * dB

            if (use_prox_l1):
                self.Weights[np.abs(self.Weights) < self.LearningRate * self.Lambda1] = 0

            if 'loss' in Log:
                Losses.append(Loss)

            if Stopfunc is not None:
                if Stopfunc(Loss):
                    break

        if 'loss' in Log:
            return Losses
        else:
            return None

    def predict(self, X):
        X = (X - self.Mean) / self.Std
        if type(X) == np.ndarray:
            return self.Sigmoid(X @ self.Weights + self.Bias)
        elif type(X) == list:
            return self.Sigmoid(np.array(X) @ self.Weights + self.Bias)
        elif type(X) == int or type(X) == float:
            return self.Sigmoid(X * self.Weights + self.Bias)


class _WithNorm:
    WithNorm = False
    Mean = None
    Std = None

    def _fit_norm(self, X):
        if self.WithNorm:
            self.Mean = np.mean(X, axis = 0)
            self.Std = np.std(X, axis = 0)
            X = (X - self.Mean) / self.Std
        return X

    def _pred_norm(self, X):
        if self.WithNorm:
            X = (X - self.Mean) / self.Std
        return X

class Sklearn_Lasso(_WithNorm, sklearn.linear_model.Lasso):
    def __init__(self, Lambda1 = 0.01, MaxIters = 1000, WithNorm = False):
        super().__init__(
            alpha = Lambda1 / 2,
            max_iter = MaxIters,
        )
        self.Lambda1 = Lambda1
        self.MaxIters = MaxIters
        self.WithNorm = WithNorm

    def fit(self, X, Y):
        super().fit(self._fit_norm(X), Y)
        return self

    def predict(self, X):
        return super().predict(self._pred_norm(X))

    def LossFunc(self, Y_pred, Y):
        Mse_Loss = np.mean((Y_pred - Y) ** 2)
        return Mse_Loss

class Sklearn_Ridge(_WithNorm, sklearn.linear_model.Ridge):
    def __init__(self, Lambda2 = 0.01, MaxIters = 1000, WithNorm = False):
        super().__init__(
            alpha = 0.0,
            max_iter = MaxIters,
        )
        self.Lambda2 = Lambda2
        self.MaxIters = MaxIters
        self.WithNorm = WithNorm

    def fit(self, X, Y):
        X = self._fit_norm(X)
        self.alpha = X.shape[0] * self.Lambda2
        super().fit(X, Y)
        return self

    def predict(self, X):
        return super().predict(self._pred_norm(X))

    def LossFunc(self, Y_pred, Y):
        Mse_Loss = np.mean((Y_pred - Y) ** 2)
        return Mse_Loss

class Sklearn_Logistic(_WithNorm, sklearn.linear_model.LogisticRegression):
    def __init__(self, C = 1.0, MaxIters = 2000, WithNorm = False, ClassWeight = None):
        super().__init__(
            C = C,
            max_iter = MaxIters,
            solver = 'lbfgs',
            class_weight = ClassWeight,
        )
        self.C = C
        self.MaxIters = MaxIters
        self.WithNorm = WithNorm
        self.ClassWeight = ClassWeight

    def fit(self, X, Y):
        super().fit(self._fit_norm(X), Y)
        return self

    def predict(self, X):
        return self.decision_function(self._pred_norm(X))

    def predict_proba(self, X):
        return super().predict_proba(self._pred_norm(X))

    def LossFunc(self, Y_pred, Y):
        Bce_Loss = np.mean(Y * np.logaddexp(0.0, -Y_pred) + (1 - Y) * np.logaddexp(0.0, Y_pred))
        return Bce_Loss

def to_raw(W, B, Mean, Std):
    W = np.asarray(W, np.float64) / Std
    return W, np.asarray(B, np.float64) - W @ Mean

def cv_search(Params, Make, X, Y, Folds, Metric, Xv = None, Yv = None):
    Cv, Va, Fits = [], [], []
    for p in Params:
        Res = np.zeros(Y.shape, dtype = float)
        for tr, te in Folds.split(X, Y):
            m = Make(p)
            m.fit(X[tr], Y[tr])
            Res[te] = m.predict(X[te])
        m = Make(p)
        m.fit(X, Y)
        Cv.append(Metric(Y, Res))
        Fits.append(m)
        if Xv is not None:
            Va.append(Metric(Yv, m.predict(Xv)))
            print(f"  {p:.3g} | cv {Cv[-1]:.6g} | valid {Va[-1]:.6g}", flush = True)
        else:
            print(f"  {p:.3g} | cv {Cv[-1]:.6g}", flush = True)
    return Cv, Va, Fits

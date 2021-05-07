import numpy as np

class SimpleLinearRegression :
    def __init__(self, fit_intercept= True):                
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept_= fit_intercept
        self.rmse_ = None

    def fit(self, X, y) :
        def sample_covariance(X, X_mean, y, y_mean) :
            cov_Xy = 0.0
            for i in range(len(X)):
                cov_Xy += (X[i] - X_mean) * (y[i] - y_mean)  
            return cov_Xy

        def sample_variance(datapoints) :
            m = np.mean(datapoints)
            var_X = 0.0
            for i in range(len(datapoints)):
                var_X += (datapoints[i] - m) ** 2
            return var_X
        
        def without_intercept(X, y) :
            nom = 0.0
            denom = 0.0
            for i in range(len(X)):
                nom += X[i] * y[i]
                denom += X[i] * X[i]
            return round(nom/denom, 2)

        if self.fit_intercept_ :
            X_mean = np.mean(X)
            y_mean = np.mean(y)

            cov_xy = sample_covariance(X, X_mean, y, y_mean)
            var_x = sample_variance(X)
            
            parameters = (round(cov_xy/var_x, 2), round(y_mean - (cov_xy/var_x) * X_mean, 2))
            
            se = 0
            for i in range(len(X)) : 
                se += (y[i] - parameters[1] - parameters[0] * X[i]) ** 2

            self.coef_ = parameters[0]
            self.intercept_ = parameters[1]
            self.rmse_ = round(np.sqrt(se/len(X)), 2)
        
        else :
            slope = without_intercept(X, y)
            se = 0
            for i in range(len(X)) : 
                se += (y[i] - slope * X[i]) ** 2

            self.coef_ = slope
            self.rmse_ = round(np.sqrt(se/len(X)), 2)

    def predict(self, X) :
        if self.fit_intercept_ :
            return self.intercept_ + np.dot(X, self.coef_)
        else :
            return np.dot(X, self.coef_)





"""An improved scikit-learn-like interface of XGBoost


About
-----
scikit-learn is machine learning library for Python.

XGBoost is a new and useful gradient boosting library,
which provides a customized Python interface as well as 
a simplified scikit-learn-like interface.

This repo contains a slightly improved and customized
scikit-learn-like interface of XGBoost, heavily based on
the official codes, with some small modifications.


Installation
------------
Install scikit-learn and xgboost, 
download this repo and place it into your projects.


License
-------
The code in this repo follows Apache License version 2.0.

scikit-learn follows New BSD License.

XGBoost follows Apache License version 2.0.


Reference
---------
[1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    http://scikit-learn.org/stable/

[2] XGBoost: eXtreme Gradient Boosting
    https://github.com/dmlc/xgboost

"""



from __future__ import print_function, division

from tempfile import NamedTemporaryFile
import os

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

import xgboost as xgb


__all__ = ['XGBEstimator', 'XGBClassifier', 'XGBRegressor']



class XGBEstimator(BaseEstimator):
    """An interface of xgboost for good intention.

    Parameters
    ----------
    n_iter : integer, optional (default=1000)
        The num_boost_round in original xgboost.

    n_jobs : integer, optional (default=-1)
        The nthread in original xgboost.
        When it is -1, the estimator would use all cores on the machine.

    learning_rate : float, optional (dafault=0.3)
        The eta in original xgboost.

    gamma : float, optional (default=0)

    max_depth : int, optional (default=6)

    min_child_weight : int, optional (default=1)

    max_delta_step : float, optional (default=0)

    subsample : float, optional (default=1)
        It ranges in (0, 1].

    colsample_bytree : float, optional (default=1)
        It ranges in (0, 1].

    base_score : float, optional (default=0.5)

    random_state : int or None, optional (default=None)

    early_stopping_rounds : int, optional (default=100)

    num_parallel_tree : int or None, optional (default=None)
        This is in experience. If it is set to an int, n_estimators would be set to 0 internally.

    verbose : bool, optional (default=True)

    objective : string, optional (default='reg:linear')

    eval_metric : string, optional (default='rmse')

    **kwargs : optional
        one possible value: num_class

    Attributes
    ----------
    bst_ : the xgboost boosted object

    """

    def __init__(self,
        n_iter = 1000,
        n_jobs=-1, 
        learning_rate=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        base_score=0.5,
        random_state=None,
        early_stopping_rounds=100,
        num_parallel_tree = None,
        verbose=True,
        objective='reg:linear',
        eval_metric='rmse',
        **kwargs):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.base_score = base_score
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.num_parallel_tree = num_parallel_tree
        self.verbose = verbose
        self.objective = objective
        self.eval_metric = eval_metric
        for parameter, value in kwargs.items():
            self.setattr(parameter, value)


    def __getstate__(self):
        # can't pickle ctypes pointers so save bst_ directly
        this = self.__dict__.copy()  # don't modify in place

        # delete = False for x-platform compatibility
        # https://bugs.python.org/issue14243
        with NamedTemporaryFile(mode="wb", delete=False) as tmp:
            this["bst_"].save_model(tmp.name)
            tmp.close()
            booster = open(tmp.name, "rb").read()
            os.remove(tmp.name)
        this.update({"bst_": booster})

        return this


    def __setstate__(self, state):
        with NamedTemporaryFile(mode="wb", delete=False) as tmp:
            tmp.write(state["bst_"])
            tmp.close()
            booster = xgb.Booster(model_file=tmp.name)
            os.remove(tmp.name)

        state["bst_"] = booster
        self.__dict__.update(state)


    def get_xgb_params(self):
        """Get the params for xgboost

        Returns
        -------
        xgb_params : dict
            The suitable params for xgboost.

        """
        xgb_params = {
            'eta': self.learning_rate,
            'gamma': self.gamma,
            'max_depth': int(self.max_depth),
            'min_child_weight': int(self.min_child_weight),
            'max_delta_step': self.max_delta_step,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'early_stopping_rounds': int(self.early_stopping_rounds),
            'objective': self.objective,
            'eval_metric': self.eval_metric
        }

        if not self.verbose:
            xgb_params['silent'] = 1

        if self.random_state is None:
            xgb_params['seed'] = np.random.randint(0, 2**32)
        else:
            xgb_params['seed'] = int(self.random_state)

        if self.n_jobs > 0:
            xgb_params['nthread'] = int(self.n_jobs)

        if hasattr(self, 'num_class'):
            xgb_params['num_class'] = int(self.num_class)

        if not (self.num_parallel_tree is None):
            # then we are using random forest!
            self.n_iter = 1
            xgb_params['num_parallel_tree'] = int(self.num_parallel_tree)

        return xgb_params


    def _ready_to_fit(self, X, y):
        """do nothing in BaseEstimator"""
        return X, y


    def fit(self, X, y, X_valid=None, y_valid=None, sample_weight=None):
        """Fit training dafa.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.

        y : array-like, shape=(n_samples,)

        X_valid : array-like or sparse matrix, shape=(n_valid_samples, n_features)
            The validation samples.

        y_valid : array-like, shape=(n_valid_samples,)

        sample_weight : array-like, shape = [n_samples], optional


        Returns
        -------
        self : object
            Returns self.

        """
        X, y = self._ready_to_fit(X, y)
        xgb_params = self.get_xgb_params()

        # xgboost accepts dense, csc, csr
        if isinstance(X, sp.sparse.coo_matrix):
            X = X.tocsc()

        if sample_weight is not None:
            xg_train = xgb.DMatrix(X, label=y, weight=sample_weight)
        else:
            xg_train = xgb.DMatrix(X, label=y)
        watchlist = [ (xg_train,'train')]

        if not (X_valid is None):
            if isinstance(X_valid, sp.sparse.coo_matrix):
                X_valid = X_valid.tocsc()
            if sample_weight is not None:
                xg_valid = xgb.DMatrix(X_valid, label=y_valid, weight=sample_weight)
            else:
                xg_valid = xgb.DMatrix(X_valid, label=y_valid)
            watchlist = [ (xg_train,'train'), (xg_valid, 'valid') ]

        if self.verbose:
            # with watchlist
            self.bst_ = xgb.train(params=xgb_params, dtrain=xg_train, num_boost_round=int(self.n_iter), evals=watchlist, early_stopping_rounds=int(self.early_stopping_rounds))
        else:
            # without watchlist
            # early stopping is not available
            self.bst_ = xgb.train(params=xgb_params, dtrain=xg_train, num_boost_round=int(self.n_iter))

        return self


    def predict(self, X):
        """Predict y for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array
            The predicted values.

        """
        xg_test = xgb.DMatrix(X)
        y = self.bst_.predict(xg_test)
        return y



class XGBClassifier(XGBEstimator, ClassifierMixin):
    """A classifier interface of xgboost for good intention.

    Parameters
    ----------
    n_iter : integer, optional (default=1000)
        The num_boost_round in original xgboost.

    n_jobs : integer, optional (default=-1)
        The nthread in original xgboost.
        When it is -1, the estimator would use all cores on the machine.

    learning_rate : float, optional (dafault=0.3)
        The eta in original xgboost.

    gamma : float, optional (default=0)

    max_depth : int, optional (default=6)

    min_child_weight : int, optional (default=1)

    max_delta_step : float, optional (default=0)

    subsample: float, optional (default=1)
        It ranges in (0, 1].

    colsample_bytree : float, optional (default=1)
        It ranges in (0, 1].

    base_score : float, optional (default=0.5)

    random_state : int or None, optional (default=None)

    early_stopping_rounds : int, optional (default=100)

    num_parallel_tree : int or None, optional (default=None)
        This is in experience. If it is set to an int, n_estimators would be set to 0 internally.

    verbose : bool, optional (default=True)

    Attributes
    ----------
    classes_ : list of classes

    bst_ : the xgboost boosted object

    """

    def __init__(self,
        n_iter = 1000,
        n_jobs=-1, 
        learning_rate=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        base_score=0.5,
        random_state=None,
        early_stopping_rounds=100,
        num_parallel_tree=None,
        verbose=True):
        super(XGBClassifier, self).__init__(n_iter,
            n_jobs, 
            learning_rate,
            gamma,
            max_depth,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            base_score,
            random_state,
            early_stopping_rounds,
            num_parallel_tree,
            verbose,
            objective='multi:softprob',
            eval_metric='mlogloss')


    def get_xgb_params(self):
        """Get the params for xgboost

        Returns
        -------
        xgb_params : dict
            The suitable params for xgboost.

        """
        xgb_params = {
            'eta': self.learning_rate,
            'gamma': self.gamma,
            'max_depth': int(self.max_depth),
            'min_child_weight': int(self.min_child_weight),
            'max_delta_step': self.max_delta_step,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'early_stopping_rounds': int(self.early_stopping_rounds)
        }

        if not self.verbose:
            xgb_params['silent'] = 1

        if self.random_state is None:
            xgb_params['seed'] = np.random.randint(0, 2**32)
        else:
            xgb_params['seed'] = int(self.random_state)

        if self.n_jobs > 0:
            xgb_params['nthread'] = int(self.n_jobs)

        # we have to figure out the num_classes here. :-(
        # that is why we want to accept y in this function
        if hasattr(self, 'classes_'):
            num_class = len(self.classes_)
            if num_class > 2:
                xgb_params['objective'] = 'multi:softprob'
                xgb_params['eval_metric'] = 'mlogloss'
                xgb_params['num_class'] = num_class
            elif num_class == 2:
                xgb_params['objective'] = 'binary:logistic'
                xgb_params['eval_metric'] = 'auc'

        if not (self.num_parallel_tree is None):
            # then we are using random forest!
            self.n_iter = 1
            xgb_params['num_parallel_tree'] = int(self.num_parallel_tree)

        return xgb_params


    def _ready_to_fit(self, X, y):
        """
        Check out the classes of y
        """
        self.classes_, y = np.unique(y, return_inverse=True)
        return X, y


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.

        """
        p = self.predict_proba(X)
        y = self.classes_[np.argmax(p, axis=p.ndim-1)]
        return y


    def decision_function(self, X):
        """Same as predict_proba()
        """
        return self.predict_proba(X)


    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """
        xg_test = xgb.DMatrix(X)

        # multi classes
        if len(self.classes_) > 2:
            p = self.bst_.predict(xg_test)

        # 2 classes
        # p is a 1 dimensional array-like if using binary:logistic
        elif len(self.classes_) == 2:
            pred = self.bst_.predict(xg_test)
            another_pred = 1 - pred
            p = np.array([another_pred, pred]).T
        
        if p.shape[0] == 1:
            p = p[0, :]    
        return p


    def predict_log_proba(self, X):
        """Predict log of class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """
        return np.log(self.predict_proba(X))


    def score(self, X, y, sample_weight=None, score_type='auto'):
        """Returns the goodness of fit on the given test data and labels.

        In original sklearn, it returns the mean accuracy.
        But in this implemention, it is possible to choose different types: 'n_mlogloss', 'mean_acc', 'auto'

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        score_type : 'n_mlogloss' or 'auto' or 'mean_acc' (default='auto')
            If 'auto', 
                the function would return Area Under Curve if binary classification,
                or negative of multi-class log loss if more than 2 classes.
            If 'n_mlogloss', 
                the function would return the negative value of multi-class log loss.   
            If 'mean_acc', 
                the function would return the mean accuracy.

        Returns
        -------
        score : float
            The higher, the better.
        """

        if not score_type in ['n_mlogloss', 'auto', 'mean_acc']:
            score_type = 'auto'

        if score_type == 'mean_acc':
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        if score_type == 'n_mlogloss':
            # we want higher better, so use negative value
            return -log_loss(y, self.predict_proba(X), sample_weight=sample_weight)

        if score_type == 'auto':
            if len(self.classes_)==2:
                return roc_auc_score(y, self.predict_proba(X)[:, 1], sample_weight=sample_weight)
            else:
                return -log_loss(y, self.predict_proba(X), sample_weight=sample_weight)



class XGBRegressor(XGBEstimator, RegressorMixin):
    """A regressor interface of xgboost for good intention.

    Parameters
    ----------
    n_iter : integer, optional (default=1000)
        The num_boost_round in original xgboost.

    n_jobs : integer, optional (default=-1)
        The nthread in original xgboost.
        When it is -1, the estimator would use all cores on the machine.

    learning_rate : float, optional (dafault=0.3)
        The eta in original xgboost.

    gamma : float, optional (default=0)

    max_depth : int, optional (default=6)

    min_child_weight : int, optional (default=1)

    max_delta_step : float, optional (default=0)

    subsample: float, optional (default=1)
        It ranges in (0, 1].

    colsample_bytree: float, optional (default=1)
        It ranges in (0, 1].

    base_score: float, optional (default=0.5)

    random_state: int or None, optional (default=None)

    early_stopping_rounds: int, optional (default=100)

    num_parallel_tree: int or None, optional (default=None)
        This is in experience. If it is set to an int, n_estimators would be set to 0 internally.

    verbose: bool, optional (default=True)

    Attributes
    ----------
    bst_ : the xgboost boosted object

    """

    def __init__(self,
        n_iter = 1000,
        n_jobs=-1, 
        learning_rate=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        base_score=0.5,
        random_state=None,
        early_stopping_rounds=100,
        num_parallel_tree = None,
        verbose = True):
        super(XGBRegression, self).__init__(n_iter,
            n_jobs, 
            learning_rate,
            gamma,
            max_depth,
            min_child_weight,
            max_delta_step,
            subsample,
            colsample_bytree,
            base_score,
            random_state,
            early_stopping_rounds,
            num_parallel_tree,
            verbose,
            objective='reg:linear',
            eval_metric='rmse')


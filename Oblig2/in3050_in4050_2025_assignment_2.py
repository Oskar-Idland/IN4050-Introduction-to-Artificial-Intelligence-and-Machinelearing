#!/usr/bin/env python
# coding: utf-8

# # IN3050/IN4050 Mandatory Assignment 2, 2025: Supervised Learning
# ## Author: Oskar Ekeid Idland (oskarei)

# ## Introduction

# ### Rules
# 
# Before you begin the exercise, review the rules at this website: https://www.uio.no/english/studies/examinations/compulsory-activities/mn-ifi-mandatory.html , in particular the paragraph on cooperation. This is an individual assignment. You are not allowed to deliver together or copy/share source-code/answers with others. Read also the "Routines for handling suspicion of cheating and attempted cheating at the University of Oslo": https://www.uio.no/english/studies/examinations/cheating/index.html 
# We do not entirely prohibit the use of generative language models ("smart assistants" like ChatGPT, Llama, Claude or Copilot), but you must clearly acknowledge this at all times, following the UiO guidelines: https://www.uio.no/english/studies/resources/ai_student.html
# Note also that you must fully understand _all_ the parts of you submissions, even if you got some help from a generative model. This will be tested during your peer review sessions (https://www.uio.no/studier/emner/matnat/ifi/IN3050/v25/Peer%20review/).
# By submitting this assignment, you confirm that you are familiar with the rules and the consequences of breaking them.
# 
# ### Delivery
# 
# **Deadline**: Friday, March 28, 2025, 23:59
# 
# Your submission should be delivered in Devilry. You may redeliver in Devilry before the deadline, but include all files in the last delivery, as only the last delivery will be read. You are recommended to upload preliminary versions hours (or days) before the final deadline.
# 
# ### What to deliver?
# 
# You are recommended to solve the exercise in a Jupyter notebook, but you might solve it in a regular Python script if you prefer.
# 
# #### Alternative 1
# If you prefer not to use notebooks, you should deliver the code, your run results, and a PDF report where you answer all the questions and explain your work.
# 
# #### Alternative 2
# If you choose Jupyter, you should deliver the notebook. You should answer all questions and explain what you are doing in Markdown. Still, the code should be properly commented. The notebook should contain results of your runs. In addition, you should make a PDF of your solution which shows the results of the runs. (If you can't export: notebook -> latex -> pdf on your own machine, you may do this on the IFI linux machines.)
# 
# Here is a list of *absolutely necessary* (but not sufficient) conditions to get the assignment marked as passed:
# 
# - You must deliver your code (Python script or Jupyter notebook) you used to solve the assignment.
# - The code used for making the output and plots must be included in the assignment. 
# - You must include example runs that clearly shows how to run all implemented functions and methods.
# - All the code (in notebook cells or python main-blocks) must be runnable. If you have unfinished code that crashes, please comment it out and document what you think causes it to crash. 
# - You must also deliver a PDF of the code, outputs, comments and plots as explained above.
# 
# Your report/notebook should contain your name and username.
# 
# Deliver one single compressed folder (.zip, .tgz or .tar.gz) which contains your complete solution.
# 
# Important: if you weren’t able to finish the assignment, use the PDF report/Markdown to elaborate on what you’ve tried and what problems you encountered. Students who have made an effort and attempted all parts of the assignment will get a second chance even if they fail initially. This exercise will be graded PASS/FAIL.

# ### Goals of the assignment
# The goal of this assignment is to get a better understanding of supervised learning with gradient descent. It will, in particular, consider the similarities and differences between linear classifiers and multi-layer feed forward neural networks (multi-layer perceptrons, MLP) and the differences and similarities between binary and multi-class classification. 
# 
# ### Tools
# The aim of the exercises is to give you a look inside the learning algorithms. You may freely use code from the weekly exercises and the published solutions. You should not use machine learning libraries like Scikit-Learn or PyTorch, because the point of this assignment is for you to implement things from scratch. You, however, are encouraged to use tools like NumPy, Pandas and MatPlotLib, which are not ML-specific.
# 
# The given precode uses NumPy. You are recommended to use NumPy since it results in more compact code, but feel free to use pure Python if you prefer. 
# 
# If anything is unclear, do not hesitate to ask. Also, if you think some assumptions are missing, make your own and explain them!

# ## Imports

# In[28]:


import sklearn # This is only to generate a dataset
import matplotlib
import numpy as np
from tqdm import tqdm
from time import time
from numba import njit 
from numba.experimental import jitclass
from typing import Literal
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams.update({'font.size': 14, 'figure.figsize': (16, 10)}) 


# ## Datasets

# We start by making a synthetic dataset of 5000 instances and ten classes, with 500 instances in each class. (See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html regarding how the data are generated.) We choose to use a synthetic dataset---and not a set of natural occuring data---because we are mostly interested in properties of the various learning algorithms, in particular the differences between linear classifiers and multi-layer neural networks together with the difference between binary and multi-class data. In addition, we would like a dataset with instances represented with only two numerical features, so that it is easy to visualize the data. It would be rather difficult (although not impossible) to find a real-world dataset of the same nature. Anyway, you surely can use the code in this assignment for training machine learning models on real-world datasets.
# 
# When we are doing experiments in supervised learning, and the data are not already split into training and test sets, we should start by splitting the data. Sometimes there are natural ways to split the data, say training, on data from one year and testing on data from a later year, but if that is not the case, we should shuffle the data randomly before splitting. (OK, that is not necessary with this particular synthetic data set, since it is already shuffled by default by Scikit-Learn, but that will not be the case with real-world data) We should split the data so that we keep the alignment between X (features) and t (class labels), which may be achieved by shuffling the indices. We split into 60% for training, 20% for validation, and 20% for final testing. The set for final testing *must not be used* till the end of the assignment in part 3.
# 
# We fix the seed both for data set generation and for shuffling, so that we work on the same datasets when we rerun the experiments. This is done by the `random_state` argument and the `rng = np.random.RandomState(424242)`.

# In[29]:


# Generating the dataset
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[500, 500, 500, 500, 500, 500, 500, 500, 500, 500], centers=[[0,1],[4,2],[8,1],[2,0],[6,0],[3,-3],[4,-2],[0,5],[0,4],[-2,-2]], 
                  n_features=2, random_state=424242, cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5, 3.0, 1.0, 0.5, 2.5, 2.5])


# In[30]:


# Shuffling the dataset
indices = np.arange(X.shape[0])
rng = np.random.RandomState(424242)
rng.shuffle(indices)
indices[:10]


# In[31]:


# Splitting into train, dev and test
X_train = X[indices[:3000],:].astype(np.float32)
X_val = X[indices[3000:4000],:].astype(np.float32)
X_test = X[indices[4000:],:].astype(np.float32)
t_multi_train = t_multi[indices[:3000]].astype(np.float32)
t_multi_val = t_multi[indices[3000:4000]].astype(np.float32)
t_multi_test = t_multi[indices[4000:]].astype(np.float32)


# Next, we will  make a second dataset with only two classes by merging the existing labels in (X,t), so that `0-5` become the new `0` and `6-9` become the new `1`. Let's call the new set (X, t2). This will be a binary set.
# We now have two datasets:
# 
# - Binary set: `(X, t2)`
# - Multi-class set: `(X, t_multi)`

# In[32]:


t2_train = t_multi_train >= 6
t2_train = t2_train.astype("int")
t2_val = (t_multi_val >= 6).astype("int")
t2_test = (t_multi_test >= 6).astype("int")


# We can plot the two traning sets.

# In[33]:


plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title("Multi-class set")


# In[34]:


plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title("Binary set")


# # Part 1: Linear classifiers
# ### Linear regression

# We see that even the binary set (X, t2) is far from linearly separable, and we will explore how various classifiers are able to handle this. We start with linear regression with the Mean Squared Error (MSE) loss, although it is not the most widely used approach for classification tasks: but we are interested. You may make your own implementation from scratch or start with the solution to the weekly exercise set 6. We include it here with a little added flexibility.

# In[35]:


@njit
def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    X = np.concatenate((biases, X), axis  = 1) 
    return X.astype(float32)


# In[36]:


@jitclass()
class NumpyClassifier():
    """Common methods to all Numpy classifiers --- if any"""
    def __init__(self):
        ...
    def fit():
        raise NotImplementedError("Inheriting classes must implement this method")
    def predict():
        raise NotImplementedError("Inheriting classes must implement this method")


# In[37]:


from numba.experimental import jitclass
from numba.types import float64, int64, Array, float32, int32

spec = [
    ('weights', float32[:]),
    ('bias', int32),
    # ('lr', float64),
    # ('epochs', int64)
]
@jitclass(spec)
class NumpyLinRegClass():
    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, lr = 0.1, epochs=10):
        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        X_train = X_train.astype(np.float32)
        t_train = t_train.astype(np.float32)
        self.weights = np.zeros(X_train.shape[1], dtype=np.float32)
        
        # X_train_T will inherit dtype from X_train
        X_train_T = X_train.T
        
        # Convert lr to float32
        lr = np.float32(lr)
        for epoch in range(epochs):
            delta = lr / X_train.shape[0] *  X_train_T @ (X_train @ self.weights - t_train)      
            self.weights -= delta.astype(np.float32)      
    
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold
    
    def spawn(self):
        """Return a new instance of the same class"""
        return NumpyLinRegClass(self.bias)
    
class NumpyLinRegClass_old():
    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, lr: float = 0.1, epochs: int =10):
        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for epoch in range(epochs):
            weights -= lr / N *  X_train.T @ (X_train @ weights - t_train)      
    
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold


# We can train and test a first classifier (on the binary dataset).

# In[38]:


@njit()
def accuracy(predicted, gold):
    return np.mean(predicted == gold)


# In[39]:


classifier = NumpyLinRegClass()
classifier.fit(X_train, t2_train)
accuracy(classifier.predict(X_val), t2_val)


# The following is a small procedure which plots the data set together with the decision boundaries. 
# You may modify the colors and the rest of the graphics as you like.
# The procedure will also work for multi-class classifiers

# In[40]:


def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'tab10')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='tab10')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")

#    plt.show()


# In[41]:


plot_decision_regions(X_train, t2_train, classifier)


# ### Task: Tuning
# 
# The result is far from impressive. 
# Remember that a classifier which always chooses the majority class will have an accuracy of 0.6 on this data set.
# 
# Your task is to try various settings for the two training hyper-parameters, learning rate and the number of epochs, to get the best accuracy on the validation set. 
# 
# Report how the accuracy varies with the hyper-parameter settings. It it not sufficient to give the final hyperparameters. You must also show how you found then and results for alternative values you tried aout.
# 
# When you are satisfied with the result, you may plot the decision boundaries, as above.

# In[42]:


class HyperparameterTuner:
    def __init__(self, classifier: NumpyClassifier, grid_params: dict[str, np.ndarray], func_params: dict[str, np.ndarray|float|int] = None):
        """
        Initialize GridSearchCV object for hyperparameter optimization.
        
        Parameters
        ----------
        classifier : NumpyClassifier
            An instance of the classifier to optimize hyperparameters for.
        grid_params : dict[str, np.ndarray]
            A dictionary of hyperparameters with the key being the hyperparameter name 
            and the value being the array of hyperparameters.
        func_params : dict[str, np.ndarray|float|int]
            A dictionary of static parameters to pass to the classifier's fit function.
        """
        self._classifier = classifier
        self._grid_params = grid_params
        self._func_params = func_params or {}
        
        self._accuracies = None
        self._best_accuracy = None
        self._best_params = {}
        self._param_arrays = {}
        
    def search(self, X_train, t_train, X_val, t_val):
        """
        Performs grid search over the hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            The training data.
        t_train : np.ndarray
            The training targets.
        X_val : np.ndarray
            The validation data.
        t_val : np.ndarray
            The validation targets.
            
        Returns
        -------
        self : GridSearchCV
            The fitted GridSearchCV object.
        """
        # Store the data the model was trained on
        self._X_train = X_train
        self._t_train = t_train
        self._X_val = X_val
        self._t_val = t_val
        
        
        param_names  = list(self.grid_params.keys())
        param_arrays = list(self.grid_params.values())
            
        param_name1,  param_name2  = param_names
        param_array1, param_array2 = param_arrays
        
        self._param_arrays = {param_name1: param_array1, param_name2: param_array2}
        
        nx = len(param_array1)
        ny = len(param_array2)
        self._accuracies = np.zeros((ny, nx))
        
        # Function to evaluate a single set of hyperparameters
        def evaluate_hyperparams(classifier: NumpyClassifier, param1, param2):
            # Creates a fresh classifier instance as we are in parallel
            
            # Parameter dictionaries for passing keyword arguments
            func_kwarg_1 = {param_name1: param1}
            func_kwarg_2 = {param_name2: param2}
            
            # Train and evaluate
            classifier.fit(X_train=X_train, t_train=t_train, **func_kwarg_1, **func_kwarg_2, **self.func_params)
            return accuracy(classifier.predict(X_val), t_val)

        # Run grid search with outer loop sequential and inner loop parallel
        for j, param1 in enumerate(tqdm(param_array1, desc=param_name1)):
            results = Parallel(n_jobs=1)(delayed(evaluate_hyperparams)(self.classifier, param1, param2) for param2 in param_array2)
            
            for i, acc in enumerate(results):
                self._accuracies[i, j] = acc
        
        self._find_best_params()
        
        # Train the classifiers with the best hyperparameters
        self._best_classifiers = []
        for i in range(len(self.best_params[param_name1])):
            best_params = {param_name1: self.best_params[param_name1][i], param_name2: self.best_params[param_name2][i]}
            classifier = self.classifier.__class__()
            classifier.fit(X_train, t_train, **best_params, **self.func_params)
            self._best_classifiers.append(classifier)
        
    
    def _find_best_params(self):
        """Find the best hyperparameters based on validation accuracy."""
        param_names = list(self.param_arrays.keys())
        param_values = list(self.param_arrays.values())
        
        self._best_accuracy = np.max(self.accuracies)
        best_idx = np.where(self.accuracies == self.best_accuracy)
        
        best_y_idx = best_idx[0]
        best_x_idx = best_idx[1]
        
        best_x = param_values[0][best_x_idx]
        best_y = param_values[1][best_y_idx]
        
        self._best_params = {
            param_names[0]: best_x,
            param_names[1]: best_y
        }
        
    
    
    def sort_best_params(self, primary_param, descending=False):
        """
        Sort the best parameters by the specified parameter.
        
        Parameters
        ----------
        primary_param : str
            The parameter to sort by.
        descending : bool
            Whether to sort in descending order.
            
        Returns
        -------
        sorted_params : dict
            A dictionary with sorted parameter values.
        """
            
        param_names = list(self.best_params.keys())
        other_param = param_names[1] if param_names[0] == primary_param else param_names[0]
        
        primary_array = self.best_params[primary_param]
        secondary_array = self.best_params[other_param]
        
        # Get the indices that would sort the primary array
        sort_indices = np.argsort(primary_array)
        # If descending is True, reverse the order
        if descending:
            sort_indices = sort_indices[::-1]
        # Sort both arrays using the same indices
        sorted_primary = primary_array[sort_indices]
        sorted_secondary = secondary_array[sort_indices]
        
        return {primary_param: sorted_primary, other_param: sorted_secondary}
    
    
    def plot_search(self, labels: tuple[str] = None, logx=False, logy=False) -> plt.Figure:
        """
        Plots the validation accuracy for different hyperparameters.
        
        Parameters
        ----------
        labels : tuple[str] (optional)
            The labels for the x and y axes.
        logx : bool (optional)
            Whether to use logarithmic scale for x-axis. Default is False
        logy : bool (optional)
            Whether to use logarithmic scale for y-axis. Default is False
            
        Returns
        -------
        fig : plt.Figure
            The figure object.
        """
        
        # Get parameter names and arrays
        param_names = list(self.param_arrays.keys())
        x_array = self.param_arrays[param_names[0]]
        y_array = self.param_arrays[param_names[1]]
        
        # Define scaling functions
        def log10(array):
            return np.log10(array)
        
        def identity(array):
            return array
        
        # Set scaling based on parameters
        scale_x = log10 if logx else identity
        scale_y = log10 if logy else identity
        
        # Create meshgrid for contour plot
        XX, YY = np.meshgrid(scale_x(x_array), scale_y(y_array))
        
        # Create contour plot
        contour = plt.contourf(XX, YY, self.accuracies, levels=len(np.unique(self.accuracies).ravel()), cmap='PiYG')
        
        # Add colorbar
        plt.colorbar(contour, label='Validation Accuracy', format=FuncFormatter('{:.2%}'.format))
        
        # Add contour lines
        contour_lines = plt.contour(XX, YY, self.accuracies, 10, colors='white', linewidths=0.75, alpha=0.9)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
        # Get best parameters
        best_param1 = self.best_params[param_names[0]]
        best_param2 = self.best_params[param_names[1]]
        
        # Ensure we have arrays for compatibility
        if np.isscalar(best_param1):
            best_param1 = np.array([best_param1])
        if np.isscalar(best_param2):
            best_param2 = np.array([best_param2])
        
        # Create parameter string for label
        parameter_str = ''.join(f'\n({param_names[0]}={x:.3g}, {param_names[1]}={y:.3g})' 
                            for x, y in zip(best_param1, best_param2))
        
        # Plot first best hyperparameter with label
        plt.scatter(scale_x(best_param1[0]), scale_y(best_param2[0]), color='hotpink', s=200, marker='*',
                edgecolors='white', linewidths=1.5, zorder=5,
                label=f'Highest: {self.best_accuracy:.2%}' + parameter_str)
        
        # Plot additional best hyperparameters if there are multiple
        for param1, param2 in zip(best_param1[1:], best_param2[1:]):
            plt.scatter(scale_x(param1), scale_y(param2), color='hotpink', s=200, marker='*',
                    edgecolors='white', linewidths=1.5, zorder=5)
        
        # Add labels
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        else:
            plt.xlabel(param_names[0] + r" [log$_{10}$]"*logx)
            plt.ylabel(param_names[1] + r" [log$_{10}$]"*logy)
        
        # Format ticks for log scales
        if logx:
            plt.xticks([np.log10(param) for param in x_array], 
                    [f'{param:.1e}' for param in x_array], rotation=45)
        if logy:
            plt.yticks([np.log10(param) for param in y_array],
                    [f'{param:.1e}' for param in y_array])
        
        plt.legend()
        plt.grid(alpha=0.2)
        
        fig = plt.gcf()
        return fig
    
    def plot_regions(self, n=0) -> None:
        """
        Plot the data set (X,t) together with the decision boundary of the classifier clf
        
        Parameters
        ----------
        n : int (optional)
            The index of the best pair of hyperparameters to use. Default is 0.
        """
        assert self.best_params, "No best hyperparameters found. Run search() first."
        assert n < len(self.best_classifiers), f'Index {n} out of range. There are {len(self.best_classifiers)} sets of hyperparameters giving the best accuracies.'
        classifier = self.best_classifiers[n]
        plot_decision_regions(self.X_train, self.t_train, classifier)
        plt.show()
        
    def plot_loss(self, n=0) -> plt.Figure:
        '''
        Plots the loss for a set of the best hyperparameters.
        
        Parameters
        ----------
        n : int (optional)
            The index of the best pair of hyperparameters to use. Default is 0.
        '''
        assert self.best_params, "No best hyperparameters found. Run search() first."
        assert n < len(self.best_classifiers), f'Index {n} out of range. There are {len(self.best_classifiers)} sets of hyperparameters giving the best accuracies.'
        
        classifier = self.best_classifiers[n]
        classifier.plot_loss()
        
        fig = plt.gcf()
        return fig
    
    @property
    def classifier(self):
        """Get the classifier."""
        return self._classifier
    
    @property
    def grid_params(self):
        """Get the grid parameters."""
        return self._grid_params
    
    @property
    def func_params(self):
        """Get the function parameters."""
        return self._func_params
        
    @property
    def best_classifiers(self):
        """Get the best classifiers."""
        return self._best_classifiers
        
    @property
    def best_accuracy(self):
        """Get the best validation accuracy."""
        return self._best_accuracy
    
    @property
    def best_params(self):
        """Get the best hyperparameters."""
        return self._best_params
    
    @property
    def param_arrays(self):
        """Get the parameter arrays."""
        return self._param_arrays
    
    @property
    def accuracies(self):
        """Get the validation accuracies."""
        return self._accuracies
    
    @property
    def X_train(self):
        """Get the training data."""
        return self._X_train
    
    @property
    def t_train(self):
        """Get the training targets."""
        return self._t_train
    
    @property
    def X_val(self):
        """Get the validation data."""
        return self._X_val
    
    @property
    def t_val(self):
        """Get the validation targets."""
        return self._t_val


# In[43]:


from numba.experimental import jitclass
from numba.types import int64, DictType, unicode_type, float32, int32
from numba.typed import Dict, List
from numba import typeof

classifier_type = typeof(NumpyLinRegClass())
dict_array_type = typeof(Dict.empty(key_type=unicode_type, value_type=float32[:]))
dict_float_type = typeof(Dict.empty(key_type=unicode_type, value_type=float32))

dict_array = Dict.empty(key_type=unicode_type, value_type=float32[:])
dict_float = Dict.empty(key_type=unicode_type, value_type=float32)

spec = [
    ('_classifier', classifier_type),
    ('_grid_params', dict_array_type),
    ('_func_params', dict_float_type),
    ('_accuracies', float32[:,:]),
    ('_best_accuracy', float32),
    ('_best_params', dict_float_type),
    ('_param_arrays', dict_array_type),
    ('_X_train', float32[:,:]),
    ('_t_train', float32[:]),
    ('_X_val', float32[:,:]),
    ('_t_val', float32[:]),
    # ('_best_classifiers', List(classifier_type)),
    ('weights', float32[:]),
    ('bias', int64),
    ('lr', float32),
    ('epochs', int64)
]

# @jitclass(spec)
# class HyperparameterTunerV2:
#     def __init__(self, classifier: NumpyLinRegClass, grid_params, func_params=None):
#         """
#         Initialize GridSearchCV object for hyperparameter optimization.
        
#         Parameters
#         ----------
#         classifier : NumpyClassifier
#             An instance of the classifier to optimize hyperparameters for.
#         grid_params : dict or Dict
#             A dictionary of hyperparameters with the key being the hyperparameter name 
#             and the value being the array of hyperparameters.
#         func_params : dict or Dict
#             A dictionary of static parameters to pass to the classifier's fit function.
#         """
#         self._classifier = classifier
        
#         # Initialize empty typed dictionaries
#         dict_array = Dict.empty(key_type=unicode_type, value_type=float64[:])
#         dict_float = Dict.empty(key_type=unicode_type, value_type=float64)
#         self._grid_params = dict_array
        
        
        
        
        
        
#         self._func_params = dict_float
#         self._param_arrays = dict_array
        
        
#         # # Copy grid_params if provided
#         # if grid_params is not None:
#         #     for key in grid_params:
#         #         self._grid_params[key] = grid_params[key]
        
#         # # Copy func_params if provided
#         # if func_params is not None:
#         #     for key in func_params:
#         #         self._func_params[key] = func_params[key]
        
#         self._best_accuracy = 0.0
        
#     def search(self, X_train, t_train, X_val, t_val):
#         """
#         Performs grid search over the hyperparameters.
        
#         Parameters
#         ----------
#         X_train : np.ndarray
#             The training data.
#         t_train : np.ndarray
#             The training targets.
#         X_val : np.ndarray
#             The validation data.
#         t_val : np.ndarray
#             The validation targets.
            
#         Returns
#         -------
#         self : GridSearchCV
#             The fitted GridSearchCV object.
#         """
#         # Store the data the model was trained on
#         self._X_train = X_train
#         self._t_train = t_train
#         self._X_val = X_val
#         self._t_val = t_val
        
        
            
#         param_name1,  param_name2  = self.grid_params.keys()
#         param_array1, param_array2 = self.grid_params.values()
        
#         # Store param arrays in typed dictionary
#         # self._param_arrays = dict_array
#         self._param_arrays[param_name1] = param_array1.astype(np.double)
#         self._param_arrays[param_name2] = param_array2.astype(np.double)
        
#         nx = len(param_array1)
#         ny = len(param_array2)
#         self._accuracies = np.zeros((ny, nx))
        
#         # Function to evaluate a single set of hyperparameters
#         def evaluate_hyperparams(classifier: NumpyLinRegClass, param1, param2):
#             # Creates a fresh classifier instance as we are in parallel
            
#             if param_name1 == "lr" and param_name2 == "epochs":
#                 classifier.fit(X_train, t_train, param1, param2)
#             elif param_name1 == "epochs" and param_name2 == "lr":
#                 classifier.fit(X_train, t_train, param2, param1)
            
#             # Train and evaluate
#             return accuracy(classifier.predict(X_val), t_val)

#         # Run grid search with outer loop sequential and inner loop parallel
#         for j, param1 in enumerate(param_array1):
#             for i, param2 in enumerate(param_array2):
#                 self._accuracies[i, j] = evaluate_hyperparams(self.classifier, param1, param2)
        
#         # self._find_best_params()
        
#         # Train the classifiers with the best hyperparameters
#         # self._best_classifiers = List(classifier_type)
#         # for i in range(len(self.best_params[param_name1])):
#         #     best_params = {param_name1: self.best_params[param_name1][i], param_name2: self.best_params[param_name2][i]}
#         #     classifier = self.classifier.spawn()
#         #     if param_name1 == "lr" and param_name2 == "epochs":
#         #         classifier.fit(X_train, t_train, best_params[param_name1], best_params[param_name2])
#         #     elif param_name1 == "epochs" and param_name2 == "lr":
#         #         classifier.fit(X_train, t_train, best_params[param_name2], best_params[param_name1])

#             # self._best_classifiers.append(classifier)
        
    
#     # def _find_best_params(self):
#     #     """Find the best hyperparameters based on validation accuracy."""
#     #     param_name1, param_name2 = self.param_arrays.keys()
#     #     param_value1, param_value2 = self.param_arrays.values()
        
#     #     self._best_accuracy = np.max(self.accuracies)
#     #     best_idx = np.where(self.accuracies == self.best_accuracy)
        
#     #     best_y_idx = best_idx[0]
#     #     best_x_idx = best_idx[1]
        
#     #     best_x = param_value1[best_x_idx[0]]
#     #     best_y = param_value2[best_y_idx[0]]
        
#     #     # Create best_params as a typed dictionary
#     #     self._best_params = dict_float
#     #     self._best_params[param_name1] = best_x
#     #     self._best_params[param_name2] = best_y
        
    
#     # def sort_best_params(self, primary_param, descending=False):
#     #     """
#     #     Sort the best parameters by the specified parameter.
        
#     #     Parameters
#     #     ----------
#     #     primary_param : str
#     #         The parameter to sort by.
#     #     descending : bool
#     #         Whether to sort in descending order.
            
#     #     Returns
#     #     -------
#     #     sorted_params : dict
#     #         A dictionary with sorted parameter values.
#     #     """
            
#     #     param_names = list(self.best_params.keys())
#     #     other_param = param_names[1] if param_names[0] == primary_param else param_names[0]
        
#     #     primary_array = self.best_params[primary_param]
#     #     secondary_array = self.best_params[other_param]
        
#     #     # Get the indices that would sort the primary array
#     #     sort_indices = np.argsort(primary_array)
#     #     # If descending is True, reverse the order
#     #     if descending:
#     #         sort_indices = sort_indices[::-1]
#     #     # Sort both arrays using the same indices
#     #     sorted_primary = primary_array[sort_indices]
#     #     sorted_secondary = secondary_array[sort_indices]
        
#     #     # Create result as a typed dictionary
#     #     result = dict_array
#     #     result[primary_param] = sorted_primary
#     #     result[other_param] = sorted_secondary
#     #     return result
    
#     # def plot_search(self, labels: tuple[str] = None, logx=False, logy=False) -> plt.Figure:
#     #     """
#     #     Plots the validation accuracy for different hyperparameters.
        
#     #     Parameters
#     #     ----------
#     #     labels : tuple[str] (optional)
#     #         The labels for the x and y axes.
#     #     logx : bool (optional)
#     #         Whether to use logarithmic scale for x-axis. Default is False
#     #     logy : bool (optional)
#     #         Whether to use logarithmic scale for y-axis. Default is False
            
#     #     Returns
#     #     -------
#     #     fig : plt.Figure
#     #         The figure object.
#     #     """
        
#     #     # Get parameter names and arrays
#     #     param_names = list(self.param_arrays.keys())
#     #     x_array = self.param_arrays[param_names[0]]
#     #     y_array = self.param_arrays[param_names[1]]
        
#     #     # Define scaling functions
#     #     def log10(array):
#     #         return np.log10(array)
        
#     #     def identity(array):
#     #         return array
        
#     #     # Set scaling based on parameters
#     #     scale_x = log10 if logx else identity
#     #     scale_y = log10 if logy else identity
        
#     #     # Create meshgrid for contour plot
#     #     XX, YY = np.meshgrid(scale_x(x_array), scale_y(y_array))
        
#     #     # Create contour plot
#     #     contour = plt.contourf(XX, YY, self.accuracies, levels=len(np.unique(self.accuracies).ravel()), cmap='PiYG')
        
#     #     # Add colorbar
#     #     plt.colorbar(contour, label='Validation Accuracy', format=FuncFormatter('{:.2%}'.format))
        
#     #     # Add contour lines
#     #     contour_lines = plt.contour(XX, YY, self.accuracies, 10, colors='white', linewidths=0.75, alpha=0.9)
#     #     plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
#     #     # Get best parameters
#     #     best_param1 = self.best_params[param_names[0]]
#     #     best_param2 = self.best_params[param_names[1]]
        
#     #     # Ensure we have arrays for compatibility
#     #     if np.isscalar(best_param1):
#     #         best_param1 = np.array([best_param1])
#     #     if np.isscalar(best_param2):
#     #         best_param2 = np.array([best_param2])
        
#     #     # Create parameter string for label
#     #     parameter_str = ''.join(f'\n({param_names[0]}={x:.3g}, {param_names[1]}={y:.3g})' 
#     #                         for x, y in zip(best_param1, best_param2))
        
#     #     # Plot first best hyperparameter with label
#     #     plt.scatter(scale_x(best_param1[0]), scale_y(best_param2[0]), color='hotpink', s=200, marker='*',
#     #             edgecolors='white', linewidths=1.5, zorder=5,
#     #             label=f'Highest: {self.best_accuracy:.2%}' + parameter_str)
        
#     #     # Plot additional best hyperparameters if there are multiple
#     #     for param1, param2 in zip(best_param1[1:], best_param2[1:]):
#     #         plt.scatter(scale_x(param1), scale_y(param2), color='hotpink', s=200, marker='*',
#     #                 edgecolors='white', linewidths=1.5, zorder=5)
        
#     #     # Add labels
#     #     if labels:
#     #         plt.xlabel(labels[0])
#     #         plt.ylabel(labels[1])
#     #     else:
#     #         plt.xlabel(param_names[0] + r" [log$_{10}$]"*logx)
#     #         plt.ylabel(param_names[1] + r" [log$_{10}$]"*logy)
        
#     #     # Format ticks for log scales
#     #     if logx:
#     #         plt.xticks([np.log10(param) for param in x_array], 
#     #                 [f'{param:.1e}' for param in x_array], rotation=45)
#     #     if logy:
#     #         plt.yticks([np.log10(param) for param in y_array],
#     #                 [f'{param:.1e}' for param in y_array])
        
#     #     plt.legend()
#     #     plt.grid(alpha=0.2)
        
#     #     fig = plt.gcf()
#     #     return fig
    
#     # def plot_regions(self, n=0) -> None:
#     #     """
#     #     Plot the data set (X,t) together with the decision boundary of the classifier clf
        
#     #     Parameters
#     #     ----------
#     #     n : int (optional)
#     #         The index of the best pair of hyperparameters to use. Default is 0.
#     #     """
#     #     assert self.best_params, "No best hyperparameters found. Run search() first."
#     #     assert n < len(self.best_classifiers), f'Index {n} out of range. There are {len(self.best_classifiers)} sets of hyperparameters giving the best accuracies.'
#     #     classifier = self.best_classifiers[n]
#     #     plot_decision_regions(self.X_train, self.t_train, classifier)
#     #     plt.show()
        
#     # def plot_loss(self, n=0) -> plt.Figure:
#     #     '''
#     #     Plots the loss for a set of the best hyperparameters.
        
#     #     Parameters
#     #     ----------
#     #     n : int (optional)
#     #         The index of the best pair of hyperparameters to use. Default is 0.
#     #     '''
#     #     assert self.best_params, "No best hyperparameters found. Run search() first."
#     #     assert n < len(self.best_classifiers), f'Index {n} out of range. There are {len(self.best_classifiers)} sets of hyperparameters giving the best accuracies.'
        
#     #     classifier = self.best_classifiers[n]
#     #     classifier.plot_loss()
        
#     #     fig = plt.gcf()
#     #     return fig
    
#     @property
#     def classifier(self):
#         """Get the classifier."""
#         return self._classifier
    
#     @property
#     def grid_params(self):
#         """Get the grid parameters."""
#         return self._grid_params
    
#     @property
#     def func_params(self):
#         """Get the function parameters."""
#         return self._func_params
        
#     # @property
#     # def best_classifiers(self):
#     #     """Get the best classifiers."""
#     #     return self._best_classifiers
        
#     @property
#     def best_accuracy(self):
#         """Get the best validation accuracy."""
#         return self._best_accuracy
    
#     @property
#     def best_params(self):
#         """Get the best hyperparameters."""
#         return self._best_params
    
#     @property
#     def param_arrays(self):
#         """Get the parameter arrays."""
#         return self._param_arrays
    
#     @property
#     def accuracies(self):
#         """Get the validation accuracies."""
#         return self._accuracies
    
#     @property
#     def X_train(self):
#         """Get the training data."""
#         return self._X_train
    
#     @property
#     def t_train(self):
#         """Get the training targets."""
#         return self._t_train
    
#     @property
#     def X_val(self):
#         """Get the validation data."""
#         return self._X_val
    
#     @property
#     def t_val(self):
#         """Get the validation targets."""
#         return self._t_val


# In[44]:

spec = [
    ('_classifier', classifier_type),
    ('_grid_params', dict_array_type),
    ('_func_params', dict_float_type),
    ('_accuracies', float32[:,:]),
    ('_best_accuracy', float32),
    ('_param_arrays', dict_array_type),
    ('_X_train', float32[:,:]),
    ('_t_train', float32[:]),
    ('_X_val', float32[:,:]),
    ('_t_val', float32[:])
]

@jitclass(spec)
class HyperparameterTunerV2:
    def __init__(self, classifier, grid_params, func_params=None):
        self._classifier = classifier
        self._grid_params = Dict.empty(key_type=unicode_type, value_type=float32[:])
        self._func_params = Dict.empty(key_type=unicode_type, value_type=float32)
        self._param_arrays = Dict.empty(key_type=unicode_type, value_type=float32[:])
        
        # Copy grid parameters
        for key in grid_params.keys():
            self._grid_params[key] = grid_params[key]
        
        self._best_accuracy = float32(0.0)
        
    def search(self, X_train, t_train, X_val, t_val):
        # Store data
        self._X_train = X_train
        self._t_train = t_train
        self._X_val = X_val 
        self._t_val = t_val
        
        # Extract parameter names and values
        param_keys = list(self._grid_params.keys())
        param_name1 = param_keys[0]
        param_name2 = param_keys[1]
        param_array1 = self._grid_params[param_name1]
        param_array2 = self._grid_params[param_name2]
        
        # Store param arrays
        self._param_arrays[param_name1] = param_array1
        self._param_arrays[param_name2] = param_array2
        
        # Create results array
        nx = len(param_array1)
        ny = len(param_array2)
        self._accuracies = np.zeros((ny, nx))
        
        # Search loop
        for j in range(nx):
            param1 = param_array1[j]
            for i in range(ny):
                param2 = param_array2[i]
                
                # Create new classifier
                clf = self._classifier.spawn()
                
                # Train with appropriate parameters
                if param_name1 == "lr" and param_name2 == "epochs":
                    clf.fit(X_train, t_train, float32(param1), float32(param2))
                else:
                    clf.fit(X_train, t_train, float32(param2), float32(param1))
                
                # Evaluate
                self._accuracies[i, j] = accuracy(clf.predict(X_val), t_val)


epochs = np.arange(100, 1100, 100).astype(np.double)
lrs = np.logspace(-3, 0, 10)

grid_params = {
    "lr": lrs,
    "epochs": epochs
}

tuner_old = HyperparameterTuner(
    classifier=NumpyLinRegClass_old(),
    grid_params=grid_params,
)

# Create compatible Numba typed dictionary
from numba import jit, njit, types
# Initialize dictionary with the correct types
grid_params_numba = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:])
# Convert numpy arrays to the correct type and add to dictionary
grid_params_numba["lr"] = lrs.astype(np.float32)
grid_params_numba["epochs"] = epochs.astype(np.float32)

NumpyLinRegClass().fit(X_train, t2_train, 0.1, 10) 

tuner_new = HyperparameterTunerV2(
    classifier=NumpyLinRegClass(),
    grid_params=grid_params_numba,
)


# In[ ]:


start = time()
# tuner_old.search(X_train, t2_train, X_val, t2_val)
end = time()
print(f"Old version took {end - start:.2f} seconds")


# In[ ]:


start = time()
tuner_new.search(X_train.astype(np.float32), t2_train.astype(np.float32), X_val.astype(np.float32), t2_val.astype(np.float32))
end = time()
print(f"New version took {end - start:.2f} seconds")

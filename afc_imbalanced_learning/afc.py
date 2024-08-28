import numpy as np
from sklearn.svm import SVC
from .conformal_transformation import conformal_transform_kernel, calculate_tau_squared
from .distance import hyperspace_l2_distance_squared
from .kernel import laplacian_kernel


class AFSCTSvm:

    def __init__(
        self, 
        C=1, 
        class_weight="balanced", 
        neg_eta=None, 
        pos_eta=None, 
        kernel=None, 
        ignore_outlier_svs=True, 
        probability=False, 
        random_state=0
    ):
        self.C = C
        self.class_weight = class_weight
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.neg_eta = neg_eta
        self.pos_eta = pos_eta

        if self.kernel is None:
            self.kernel = laplacian_kernel

        self.ignore_outlier_svs = ignore_outlier_svs

    def fit(self, X, y, sample_weight=None):
        # main training function

        # compute kernel transformation
        computed_conformal_transform_kernel = self.fit_conformal_transformed_kernel(X, y, sample_weight)
        
        #fit SVM with modified kernel (computed_conformal_transform_kernel)
        if sample_weight is not None: # set class weight to None when using sample weights
            self.svm = SVC(
                C=self.C,
                kernel="precomputed",
                probability=self.probability,
                random_state=self.random_state
            )
            self.svm.fit(computed_conformal_transform_kernel, self.y_train, sample_weight=sample_weight)
        else:
        
            self.svm = SVC(
                C=self.C,
                class_weight=self.class_weight,
                kernel="precomputed",
                probability=self.probability,
                random_state=self.random_state

            )
            self.svm.fit(computed_conformal_transform_kernel, self.y_train)

    def fit_conformal_transformed_kernel(self, X, y, sample_weight=None):

        #main logic to compute kernel transformation
        

        self.X_train = X
        self.y_train = y
        
        #fit normal SVM first to obtained support vectors

        if sample_weight is not None:
            self.svm = SVC(
                C=self.C,
                kernel="precomputed",
                probability=self.probability,
                random_state=self.random_state

            )

            computed_kernel = self.kernel(self.X_train, self.X_train)
            self.svm.fit(computed_kernel, self.y_train, sample_weight=sample_weight)

        else:


            self.svm = SVC(
                C=self.C,
                class_weight=self.class_weight,
                kernel="precomputed",
                probability=self.probability,
                random_state=self.random_state

            )

            computed_kernel = self.kernel(self.X_train, self.X_train)
            self.svm.fit(computed_kernel, self.y_train)

        # extract support vectors from trained normal SVM
        # output should be an array of data points that are considered as support vectors
        support_vectors_pos, support_vectors_neg = (
            self.extract_separate_support_vectors()
        )
        
        #compute eta param for negative class according to the paper
        #eta for positive class needed to be pre-defined anyway
        if self.neg_eta == "auto":
            self.neg_eta = self.pos_eta * len(support_vectors_neg)/len(support_vectors_pos)

        #computed tau value according to the paper
        tau_squareds = self.calculate_tau_squared()
        support_vectors = np.vstack((support_vectors_pos, support_vectors_neg))

        self.tau_squareds = tau_squareds
        self.support_vectors = support_vectors

        #after we have tau and support vectors location we can now calculate D(X)
        #we then compute new Kernel using K'(Xi, Xj) = D(Xi)D(Xk)K(Xi, Xj)
        computed_conformal_transform_kernel = conformal_transform_kernel(
            self.X_train, self.X_train, computed_kernel, support_vectors, tau_squareds
        )

        return computed_conformal_transform_kernel
    
    def get_computed_conformal_transform_kernel(self, X):
        computed_kernel = self.kernel(X, self.X_train)
        computed_conformal_transform_kernel = conformal_transform_kernel(
            X, self.X_train, computed_kernel, self.support_vectors, self.tau_squareds
        )
        return computed_conformal_transform_kernel

    def predict(self, X):
        computed_conformal_transform_kernel = self.get_computed_conformal_transform_kernel(X)
        return self.svm.predict(computed_conformal_transform_kernel)

    def predict_proba(self, X):
        computed_conformal_transform_kernel = self.get_computed_conformal_transform_kernel(X)
        return self.svm.predict_proba(computed_conformal_transform_kernel)
    
    def decision_function(self, X):
        computed_kernel = self.kernel(X, self.X_train)
        computed_conformal_transform_kernel = conformal_transform_kernel(
            X, self.X_train, computed_kernel, self.support_vectors, self.tau_squareds
        )
        return self.svm.decision_function(computed_conformal_transform_kernel)

    def extract_separate_support_vectors(self):

        # extract support vector for SVM

        support_vectors = self.X_train[self.svm.support_]
        support_vectors_class = self.y_train[self.svm.support_]
        support_vectors_pos = support_vectors[np.where(support_vectors_class == 1)]
        support_vectors_neg = support_vectors[np.where(support_vectors_class == 0)]

        # if ignore outlier support vectors is set to true
        # we will ignore support vector that are misclassified from the initial trained SVM
        if self.ignore_outlier_svs:
            
            support_vectors_pos_pred = self.svm.predict(
                self.kernel(support_vectors_pos, self.X_train)
            )
            # filter only support vectors that have correctly predicted only
            support_vectors_pos = support_vectors_pos[support_vectors_pos_pred == 1]

            support_vectors_neg_pred = self.svm.predict(
                self.kernel(support_vectors_neg, self.X_train)
            )
            support_vectors_neg = support_vectors_neg[support_vectors_neg_pred == 0]

        self.support_vectors_pos = support_vectors_pos
        self.support_vectors_neg = support_vectors_neg

        return support_vectors_pos, support_vectors_neg

    def calculate_tau_squared(self):
        #function to compute tau value according to the paper
        print(f'calculating tau with pos eta: {self.pos_eta} and neg eta: {self.neg_eta}')
        
        #compute rieman distance for each pairs of positive and negative support vectors
        #out put should be a matrix with each row i and column j denotes
        #rieman distance in kernel space from ith positive vector to kth positive vector
        distances = hyperspace_l2_distance_squared(
            self.support_vectors_pos, self.support_vectors_neg, self.kernel
        )
        tau_squared_pos = calculate_tau_squared(distances, self.pos_eta)
        tau_squared_neg = calculate_tau_squared(distances.T, self.neg_eta)
        tau_squareds = np.concatenate((tau_squared_pos, tau_squared_neg))
        return tau_squareds

import numpy as np
from sklearn.svm import SVC
from .conformal_transformation import conformal_transform_kernel, calculate_tau_squared
from .distance import hyperspace_l2_distance_squared
from .kernel import laplacian_kernel


class AFSCTSvm:

    def __init__(
        self, C=1, class_weight="balanced", kernel=None, ignore_outlier_svs=True
    ):
        self.C = C
        self.class_weight = class_weight
        self.kernel = kernel

        if self.kernel is None:
            self.kernel = laplacian_kernel

        self.ignore_outlier_svs = ignore_outlier_svs

    def fit(self, X, y, sample_weight=None):
        computed_conformal_transform_kernel = self.fit_conformal_transformed_kernel(X, y, sample_weight)

        if sample_weight is not None:
            self.svm = SVC(
                C=self.C,
                kernel="precomputed", # set class weight to None when using sample weights
                probability=False,
            )
            self.svm.fit(computed_conformal_transform_kernel, self.y_train, sample_weight=sample_weight)
        else:
        
            self.svm = SVC(
                C=self.C,
                class_weight=self.class_weight,
                kernel="precomputed",
                probability=False,
            )
            self.svm.fit(computed_conformal_transform_kernel, self.y_train)

    def fit_conformal_transformed_kernel(self, X, y, sample_weight=None):
        self.X_train = X
        self.y_train = y

        if sample_weight is not None:
            self.svm = SVC(
                C=self.C,
                kernel="precomputed",
                probability=False,
            )

            computed_kernel = self.kernel(self.X_train, self.X_train)
            self.svm.fit(computed_kernel, self.y_train, sample_weight=sample_weight)

        else:


            self.svm = SVC(
                C=self.C,
                class_weight=self.class_weight,
                kernel="precomputed",
                probability=False,
            )

            computed_kernel = self.kernel(self.X_train, self.X_train)
            self.svm.fit(computed_kernel, self.y_train)

        support_vectors_pos, support_vectors_neg = (
            self.extract_separate_support_vectors()
        )
        tau_squareds = self.calculate_tau_squared()
        support_vectors = np.vstack((support_vectors_pos, support_vectors_neg))

        self.tau_squareds = tau_squareds
        self.support_vectors = support_vectors

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
        support_vectors = self.X_train[self.svm.support_]
        support_vectors_class = self.y_train[self.svm.support_]
        support_vectors_pos = support_vectors[np.where(support_vectors_class == 1)]
        support_vectors_neg = support_vectors[np.where(support_vectors_class == 0)]

        if self.ignore_outlier_svs:
            
            support_vectors_pos_pred = self.svm.predict(
                self.kernel(support_vectors_pos, self.X_train)
            )
            support_vectors_pos = support_vectors_pos[support_vectors_pos_pred == 1]

            support_vectors_neg_pred = self.svm.predict(
                self.kernel(support_vectors_neg, self.X_train)
            )
            support_vectors_neg = support_vectors_neg[support_vectors_neg_pred == 0]

        self.support_vectors_pos = support_vectors_pos
        self.support_vectors_neg = support_vectors_neg
        return support_vectors_pos, support_vectors_neg

    def calculate_tau_squared(self):
        distances = hyperspace_l2_distance_squared(
            self.support_vectors_pos, self.support_vectors_neg, self.kernel
        )
        tau_squared_pos = calculate_tau_squared(distances)
        tau_squared_neg = calculate_tau_squared(distances.T)
        tau_squareds = np.concatenate((tau_squared_pos, tau_squared_neg))
        return tau_squareds

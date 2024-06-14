import unittest
import numpy as np
import pandas as pd

from afc_imbalanced_learning.afc import AFSCTSvm

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from afc_imbalanced_learning.kernel import laplacian_kernel

def calculate_gmean(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn+fp)
  sensitivity = tp / (tp+fn)
  return np.sqrt(specificity * sensitivity)

class TestAFSCTSvm(unittest.TestCase):

    def setUp(self):
        glass_identification = fetch_ucirepo(id=42)
        features = glass_identification.data.features
        targets = glass_identification.data.targets

        # Create DataFrame
        df = pd.concat((features, targets), axis=1)

        # Map target values
        df['target'] = df['Type_of_glass'].map({
            2: 0,
            1: 0,
            7: 0,
            3: 0,
            5: 1,
            6: 0,
        })

        # Drop the original target column
        df = df.drop(columns=['Type_of_glass'])

        # Define feature columns
        feature_cols = features.columns.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_cols], df['target'],
            stratify=df['target'],
            test_size=0.2,
            random_state=42
        )

        # Scale the data
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

        # Initialize the model
        self.model = AFSCTSvm(kernel=laplacian_kernel)

    def test_fit(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.svm)
        self.assertTrue(hasattr(self.model, 'support_vectors'))

    def test_predict(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_predict_proba(self):
        self.model.fit(self.X_train, self.y_train)
        probabilities = self.model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], self.X_test.shape[0])
    
    def test_svm_gmean(self):
        svm = SVC(C=1, class_weight="balanced", kernel="precomputed")
        svm.fit(laplacian_kernel(self.X_train, self.X_train), self.y_train)

        y_pred = svm.predict(laplacian_kernel(self.X_test, self.X_train))

        g_mean = calculate_gmean(self.y_test, y_pred)

        # Assert that the G-mean is around 0.806
        self.assertAlmostEqual(g_mean, 0.806, places=3)

    def test_afc_gmean(self):
        act_svm = AFSCTSvm(C=1, class_weight="balanced", kernel=laplacian_kernel)
        act_svm.fit(self.X_train, self.y_train)

        y_pred = act_svm.predict(self.X_test)

        g_mean = calculate_gmean(self.y_test, y_pred)

        # Assert that the G-mean is around 0.987
        self.assertAlmostEqual(g_mean, 0.987, places=3)

if __name__ == '__main__':
    unittest.main()
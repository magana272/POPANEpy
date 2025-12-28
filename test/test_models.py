"""
Tests for emotion.models module
"""
import unittest
import numpy as np
from emotion.models.random_forest import EmotionRandomForest


class TestEmotionRandomForest(unittest.TestCase):
    """Test EmotionRandomForest model"""
    
    def setUp(self):
        self.model = EmotionRandomForest(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.choice(['joy', 'anger', 'fear'], size=100)
        
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.choice(['joy', 'anger', 'fear'], size=20)
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.n_estimators, 10)
    
    def test_fit(self):
        """Test that model can be trained"""
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.model, 'classes_'))
    
    def test_predict(self):
        """Test that model can make predictions"""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_evaluate(self):
        """Test that model can be evaluated"""
        self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.evaluate(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_feature_importances(self):
        """Test that feature importances are returned"""
        self.model.fit(self.X_train, self.y_train)
        importances = self.model.feature_importances()
        self.assertEqual(len(importances), 10)
        self.assertTrue(all(importances >= 0))
    
    def test_save_and_load_model(self):
        """Test saving and loading model"""
        import tempfile
        import os
        
        self.model.fit(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pkl')
            self.model.save_model(filepath)
            
            # Create new model and load
            new_model = EmotionRandomForest()
            new_model.load_model(filepath)
            
            # Predictions should match
            pred1 = self.model.predict(self.X_test)
            pred2 = new_model.predict(self.X_test)
            np.testing.assert_array_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()

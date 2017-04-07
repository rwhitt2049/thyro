from unittest import TestCase

import numpy as np
from typing import Callable

from tests.testing_utils import TEST_SIGNAL
from thyro.features import create_feature, Feature


class TestFeatureFunction(TestCase):
    def setUp(self):
        self.data = TEST_SIGNAL
        self.feature = create_feature(self.data, np.mean)

    def test_returned_type(self):
        self.assertIsInstance(self.feature, Callable)

    def test_call_with_defaults(self):
        self.assertEqual(self.feature(), self.data.mean())

    def test_returned_functionality(self):
        test_slice = slice(2, 6)
        test = self.feature(test_slice)
        self.assertEqual(test, self.data[test_slice].mean())

    def test_function_not_builtin(self):
        with self.assertRaises(NotImplementedError):
            create_feature(self.data, '__test__')


class TestFeatureCallableClass(TestCase):
    def setUp(self):
        self.data = TEST_SIGNAL
        self.feature = Feature(self.data, np.mean, 'test')

    def test_returned_type(self):
        self.assertIsInstance(self.feature, Callable)

    def test_call_with_defaults(self):
        self.assertEqual(self.feature(), self.data.mean())

    def test_returned_functionality(self):
        test_slice = slice(2, 6)
        test = self.feature(test_slice)
        self.assertEqual(test, self.data[test_slice].mean())

    def test_categorical_false(self):
        self.assertFalse(self.feature.is_categorical)

    def test_categorical_true(self):
        feature = Feature(self.data, np.mean, 'test', is_categorical=True)
        self.assertTrue(feature.is_categorical)


class TestFeatureOperationTypeError(TestCase):
    def test_bad_arguement(self):
        """Test TypeError is raised if arg isn't a callable"""
        bad_call = 5
        with self.assertRaises(TypeError):
            create_feature(TEST_SIGNAL, bad_call)

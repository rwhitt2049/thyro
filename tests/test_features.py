from unittest import TestCase

import numpy as np
from typing import Callable

from tests.testing_utils import TEST_SIGNAL
from thyro.features import get_feature, Feature


class TestFeatureFunction(TestCase):
    def setUp(self):
        self.data = TEST_SIGNAL
        self.feature = get_feature(self.data, np.mean)

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
            get_feature(self.data, '__test__')


class TestFeatureCallableClass(TestCase):
    def setUp(self):
        self.data = TEST_SIGNAL
        self.feature = Feature(self.data, np.mean)

    def test_returned_type(self):
        self.assertIsInstance(self.feature, Callable)

    def test_call_with_defaults(self):
        self.assertEqual(self.feature(), self.data.mean())

    def test_returned_functionality(self):
        test_slice = slice(2, 6)
        test = self.feature(test_slice)
        self.assertEqual(test, self.data[test_slice].mean())


class TestFeatureCallableNotStrOrCallable(TestCase):
    def test_bad_arguement(self):
        """Test TypeError is raised if arg isn't a callable"""
        bad_call = 5
        with self.assertRaises(TypeError):
            get_feature(TEST_SIGNAL, bad_call)

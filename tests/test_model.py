from unittest import TestCase

import torch

from finsler.distributions import NonCentralNakagami
from finsler.gplvm import Gplvm
from finsler.utils.helper import pickle_load


# write test function so that the model load the correct arguments
class Test_model(TestCase):
    def setUp(self):
        # load previously training gplvm model
        model_saved = pickle_load(folder_path="models/qPCR/", file_name="model.pkl")
        self.model_saved = model_saved

    def test_model_keys(self):
        # check if the model has the correct keys
        self.assertIn("model", self.model_saved)
        self.assertIn("kernel", self.model_saved)
        self.assertIn("Y", self.model_saved)
        self.assertIn("X", self.model_saved)

    # check the type and format of the keys
    def test_model_type(self):
        self.assertIsInstance(self.model_saved["Y"], torch.Tensor)
        self.assertIsInstance(self.model_saved["X"], torch.Tensor)

    def test_model_shape(self):
        self.assertEqual(self.model_saved["Y"].shape, (437, 48))
        self.assertEqual(self.model_saved["X"].shape, (437, 2))

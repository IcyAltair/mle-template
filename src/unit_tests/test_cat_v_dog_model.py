import unittest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.models.CatVDogModel import CatVDogModel


class TestCatVDogPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ROOT = Path(__file__).resolve().parents[2]
        cls.model = CatVDogModel(config_path=str(ROOT / "config.ini"), show_log=False)

    def test_build_preprocess_returns_compose(self):
        p = self.model.build_preprocess()
        self.assertTrue(hasattr(p, "__call__"))

    def test_preprocess_image_shape_dtype(self):
        img = Image.new("RGB", (512, 512))
        x = self.model.preprocess_image(img)
        self.assertEqual(tuple(x.shape), (1, 3, 224, 224))
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertEqual(x.dtype, torch.float32)

    def test_preprocess_image_rgb_conversion(self):
        img = Image.new("L", (300, 300))
        x = self.model.preprocess_image(img)
        self.assertEqual(tuple(x.shape), (1, 3, 224, 224))

    def test_embed_tensor_output(self):
        img = Image.new("RGB", (512, 512))
        x = self.model.preprocess_image(img)
        emb = self.model.embed_tensor(x, device="cpu")
        self.assertTrue(isinstance(emb, np.ndarray))
        self.assertEqual(emb.ndim, 1)
        self.assertTrue(emb.shape[0] > 0)
        self.assertEqual(emb.dtype, np.float32)
        n = np.linalg.norm(emb)
        self.assertTrue(np.isfinite(n))
        self.assertTrue(abs(n - 1.0) < 1e-3)

    def test_embed_pil_equals_embed_tensor(self):
        img = Image.new("RGB", (512, 512))
        x = self.model.preprocess_image(img)
        emb1 = self.model.embed_tensor(x, device="cpu")
        emb2 = self.model.embed_pil(img, device="cpu")
        self.assertTrue(np.allclose(emb1, emb2, atol=1e-6))

    def test_set_device_cpu(self):
        self.model.set_device("cpu")
        self.assertEqual(str(self.model.device), "cpu")



if __name__ == "__main__":
    unittest.main()
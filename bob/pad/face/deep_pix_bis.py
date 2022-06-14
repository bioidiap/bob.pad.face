import logging

import numpy as np
import torch
import torchvision.transforms as vision_transforms

from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torchvision import models

from bob.extension.download import get_file
from bob.io.image import to_matplotlib

logger = logging.getLogger(__name__)


DEEP_PIX_BIS_PRETRAINED_MODELS = {
    "oulu-npu-p1": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_1_model_0_0-24844429.pth"
    ],
    "oulu-npu-p2": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_2_model_0_0-4aae2f3a.pth"
    ],
    "oulu-npu-p3-1": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_1_model_0_0-f0e70cf3.pth"
    ],
    "oulu-npu-p3-2": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_2_model_0_0-92594797.pth"
    ],
    "oulu-npu-p3-3": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_3_model_0_0-71e18149.pth"
    ],
    "oulu-npu-p3-4": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_4_model_0_0-d7f666e5.pth"
    ],
    "oulu-npu-p3-5": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_5_model_0_0-fc40ba69.pth"
    ],
    "oulu-npu-p3-6": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_3_6_model_0_0-123a6c92.pth"
    ],
    "oulu-npu-p4-1": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_1_model_0_0-5f8dc7cf.pth"
    ],
    "oulu-npu-p4-2": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_2_model_0_0-168f2644.pth"
    ],
    "oulu-npu-p4-3": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_3_model_0_0-db57e3b5.pth"
    ],
    "oulu-npu-p4-4": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_4_model_0_0-e999b7e8.pth"
    ],
    "oulu-npu-p4-5": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_5_model_0_0-dcd13b8b.pth"
    ],
    "oulu-npu-p4-6": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_OULU_Protocol_4_6_model_0_0-96a1ab92.pth"
    ],
    "replay-mobile": [
        "http://www.idiap.ch/software/bob/data/bob/bob.pad.face/deep_pix_bis_RM_grandtest_model_0_0-6761ca7e.pth"
    ],
}
"A dictionary with the url paths to pre-trained weights of the DeepPixBis model."


class DeepPixBiS(nn.Module):
    """The class defining Deep Pixelwise Binary Supervision for Face Presentation
    Attack Detection:

    Reference: Anjith George and Sébastien Marcel. "Deep Pixel-wise Binary Supervision for
    Face Presentation Attack Detection." In 2019 International Conference on Biometrics (ICB).IEEE, 2019.

    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch.
    """

    def __init__(self, pretrained=True, **kwargs):
        """
        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
        """
        super().__init__(**kwargs)

        dense = models.densenet161(pretrained=pretrained)

        features = list(dense.features.children())

        self.enc = nn.Sequential(*features[0:8])

        self.dec = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        """Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor`
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor`
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.

        """
        enc = self.enc(x)

        dec = self.dec(enc)

        dec = nn.Sigmoid()(dec)

        dec_flat = dec.view(-1, 14 * 14)

        op = self.linear(dec_flat)

        op = nn.Sigmoid()(op)

        return dec, op


class DeepPixBisClassifier(BaseEstimator, ClassifierMixin):
    """The class implementing the DeepPixBiS score computation"""

    def __init__(
        self,
        model_file=None,
        transforms=None,
        scoring_method="pixel_mean",
        device=None,
        threshold=0.8,
        **kwargs,
    ):

        """Init method

        Parameters
        ----------
        model_file: str
          The path of the trained PAD network to load or one of the keys to :py:attr:`DEEP_PIX_BIS_PRETRAINED_MODELS`
        transforms: :py:mod:`torchvision.transforms`
          Tranform to be applied on the image
        scoring_method: str
          The scoring method to be used to get the final score,
          available methods are ['pixel_mean','binary','combined'].
        threshold: float
            The threshold to be used to binarize the output of the DeepPixBiS model.
            This is not used in the normal bob.pad.base pipeline.
        """
        super().__init__(**kwargs)

        if transforms is None:
            transforms = vision_transforms.Compose(
                [
                    vision_transforms.ToTensor(),
                    vision_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # model
        self.transforms = transforms
        self.model = None
        self.scoring_method = scoring_method.lower()
        if self.scoring_method not in ("pixel_mean", "binary", "combined"):
            raise ValueError(
                "Scoring method {} is not implemented.".format(
                    self.scoring_method
                )
            )
        self.device = device
        self.threshold = threshold

        logger.debug(
            "Scoring method is : {}".format(self.scoring_method.upper())
        )

        if model_file in DEEP_PIX_BIS_PRETRAINED_MODELS:
            model_urls = DEEP_PIX_BIS_PRETRAINED_MODELS[model_file]
            filename = model_urls[0].split("/")[-1]
            file_hash = (
                model_urls[0].split("/")[-1].split("-")[-1].split(".")[0]
            )
            model_file = get_file(
                filename,
                model_urls,
                cache_subdir="models",
                file_hash=file_hash,
                extract=False,
            )

        logger.debug("Using pretrained model {}".format(model_file))
        self.model_file = model_file

    def load_model(self):
        if self.model is not None:
            return

        cp = torch.load(
            self.model_file, map_location=lambda storage, loc: storage
        )

        self.model = DeepPixBiS(pretrained=False)
        self.model.load_state_dict(cp["state_dict"])
        self.place_model_on_device()
        self.model.eval()
        logger.debug("Loaded the pretrained PAD model")

    def predict_proba(self, images):
        """Scores face images for PAD

        Parameters
        ----------
        image : 3D :py:class:`numpy.ndarray`
          The image to extract the score from. Its size must be 3x224x224;

        Returns
        -------
        output : float
          The output score is close to 1 for bonafide and 0 for PAs.
        """
        self.load_model()

        tensor_images = []
        for img in images:
            img = to_matplotlib(img)
            with torch.no_grad():
                img = self.transforms(img)
            tensor_images.append(img)

        images = tensor_images = torch.stack(tensor_images).to(self.device)

        with torch.no_grad():
            outputs = self.model.forward(images)

        output_pixel = outputs[0].cpu().detach().numpy().mean(axis=(1, 2, 3))
        output_binary = outputs[1].cpu().detach().numpy().mean(axis=1)

        score = {
            "pixel_mean": output_pixel,
            "binary": output_binary,
            "combined": (output_binary + output_pixel) / 2,
        }[self.scoring_method]

        print(score)
        return score

    def predict(self, X):
        scores = self.predict_proba(X)
        return np.int(scores > self.threshold)

    def fit(self, X, y=None):
        """No training required for this model"""
        return self

    def __getstate__(self):
        # Handling unpicklable objects
        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"requires_fit": False}

    def place_model_on_device(self):
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if self.model is not None:
            self.model.to(self.device)

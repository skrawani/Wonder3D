from sagemaker_inference import (
    content_types,
    decoder,
    inference_handler,
    encoder,
    errors,
)
import torch
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import (
    DiffusionPipeline,
)  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers


class ModelHandler:
    def __init__(self):
        self.initialized = False
        self.mx_model = None
        self.shapes = None
        self.pipeline = None

    def initialize(self, context=None):
        """Loads a model. For PyTorch, a default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.
            context (obj): the request context (default: None).

        Returns: A PyTorch model.
        """
        self.initialized = True
        self.pipeline = DiffusionPipeline.from_pretrained(
            "flamehaze1115/wonder3d-v1.0",  # or use local checkpoint './ckpts'
            custom_pipeline="flamehaze1115/wonder3d-pipeline",
            torch_dtype=torch.float16,
        )

        # enable xformers
        self.pipeline.unet.enable_xformers_memory_efficient_attention()

        if torch.cuda.is_available():
            self.pipeline.to("cuda:0")

    def preprocess(self, input_data):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
            context (obj): the request context (default: None).

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        cond = Image.open(
            requests.get(
                "https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True
            ).raw
        )
        return Image.fromarray(np.array(cond)[:, :, :3])

    def inference(self, data):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn
            context (obj): the request context (default: None).

        Returns: a prediction
        """
        
        images = self.pipeline(
            data, num_inference_steps=20, output_type="pt", guidance_scale=1.0
        ).images

        return images
        

    def postprocess(self, model_out):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
            context (obj): the request context (default: None).

        Returns: output data serialized
        """
        return make_grid(model_out, nrow=6, ncol=2, padding=0, value_range=(0, 1))

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

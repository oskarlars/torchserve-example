import io
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)

class TinyModelHandler:
    """
    TinyModel handler class.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """
        load eager mode state_dict based model
        """
        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(f"Device on initialization is: {self.device}")
        model_dir = properties.get("model_dir")

        '''manifest = ctx.manifest
        logger.error(manifest)
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model definition file")

        logger.debug(model_pt_path)'''

        from model import TinyModel

        #state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = TinyModel()
        #self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        #logger.debug("Model file {0} loaded successfully".format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
        Scales and normalizes a PIL image
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        image = Image.open(io.BytesIO(image)).convert(
            "RGB"
        )  # in case of an alpha channel
        image = image_transform(image).unsqueeze_(0)
        print(image)
        return image

    def inference(self, img):
        logger.info(f"Device on inference is: {self.device}")
        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)
        print(outputs)
        logging.debug(outputs.shape)
        return outputs

    def postprocess(self, inference_output):

        if torch.cuda.is_available():
            inference_output = inference_output.cpu()
        else:
            inference_output = inference_output

        inference_output = inference_output.squeeze(0).permute(1, 2, 0).numpy()
        inference_output = (inference_output * 255).astype(np.uint8)
        image = Image.fromarray(inference_output).copy().convert("RGB")
        
        output = io.BytesIO()
        image.save(output, format='JPEG')
        bin_img_data = output.getvalue()

        return [bin_img_data]

_service = TinyModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data

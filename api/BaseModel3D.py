import aiohttp
from io import BytesIO
from transformers.image_utils import load_image
from PIL import Image
import requests

class BaseModel3D:
    def __init__(self):
        pass

    def predict(self, image_str, mask_str=None, prompts = ""):
        raise NotImplementedError("Subclasses should implement this method")

    def read_image(self, image_url, format="http"):
        if format == "http":
            response = requests.get(url=image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        elif format == "local":
            image = Image.open(image_url)
        image = load_image(image)
        return image
import base64
import io

import onnxruntime as ort
from PIL import Image
import numpy as np

from mlflow.pyfunc.model import PythonModel

# Define your class-color map (example with 3 classes)
CLASS_COLORS = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Class 1
    2: (0, 255, 0),     # Class 2
}


class ONNXSegmentationWrapper(PythonModel):

    def load_context(self, context):
        self.session = ort.InferenceSession(context.artifacts["onnx_model"])
        self.input_name = self.session.get_inputs()[0].name

    def _decode_base64_image(self, b64_string):
        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)

    def _preprocess(self, np_image: np.ndarray) -> np.ndarray:
        # Transpose to CHW format and normalize if needed
        image = np_image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def _postprocess(self, output) -> np.ndarray:
        # Assume output shape is [1, num_classes, H, W]
        logits = output[0]  # take first (and only) batch
        prediction: np.ndarray = np.argmax(logits, axis=0)  # shape: (H, W)

        # Map class indices to RGB
        h, w = prediction.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            rgb_image[prediction == cls] = color

        return rgb_image

    def _encode_base64_image(self, rgb_image: np.ndarray) -> str:
        image = Image.fromarray(rgb_image)
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, context, model_input: list[str], params=None) -> list[str]:
        """
        model_input: DataFrame with one column 'image', each row is a base64 string
        """
        results: list[str] = []
        for b64_string in model_input:
            np_image = self._decode_base64_image(b64_string)
            input_tensor = self._preprocess(np_image)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            rgb_output = self._postprocess(outputs)
            result_b64 = self._encode_base64_image(rgb_output)
            results.append(result_b64)
        return results

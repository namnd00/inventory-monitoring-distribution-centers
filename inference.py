import json
import os
import io
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ExifTags


JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """
    Neural network architecture.
    The ResNet50 model from PyTorch's vision library is used as a feature extractor.
    A fully connected layer is added on top of the extracted features to do the classification.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5))

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


def model_fn(model_dir):
    """
    Load a trained model from disk.
    """
    model = Net().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    """
    Deserialize and preprocess the input data.
    """
    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        try:
            exif = dict(image._getexif().items())
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        return image
    elif content_type == JSON_CONTENT_TYPE:
        url = json.loads(request_body)['url']
        img_content = requests.get(url).content
        image = Image.open(io.BytesIO(img_content)).convert('RGB')
        return image
    else:
        raise Exception('Unsupported ContentType in content_type: {}'.format(content_type))


def predict_fn(input_object, model):
    """
    Run the prediction on the preprocessed input data using the loaded model.
    """
    input_tensor = transforms.ToTensor()(input_object)
    input_tensor = transforms.Resize((224, 224))(input_tensor)
    input_tensor = input_tensor.to(device).half()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
    return output.cpu().numpy().astype('float16')


def output_fn(predictions, content_type):
    """
    Serialize the model predictions and set the correct MIME type for the response payload.
    """
    if content_type == "application/json":
        return json.dumps(predictions.tolist())
    else:
        raise Exception('Unsupported ContentType in content_type: {}'.format(content_type))

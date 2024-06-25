import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights

# Define the model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:21].eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x

class GramMatrix(nn.Module):
    def forward(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

class StyleTransferModel(nn.Module):
    def __init__(self, style_img):
        super(StyleTransferModel, self).__init__()
        self.vgg = VGG().to(device)
        self.gram = GramMatrix().to(device)
        self.style_features = self.extract_features(style_img)

    def extract_features(self, img):
        features = self.vgg(img)
        style_features = [self.gram(features)]
        return style_features

    def forward(self, x):
        content_features = self.vgg(x)
        style_features = [self.gram(content_features)]
        return style_features

def load_image(img_path, transform=None):
    image = Image.open(img_path)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device)

def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    tensor = transforms.ToPILImage()(tensor)
    return np.array(tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the style image
style_image_path = 'candy.jpg'
transform = transforms.Compose([
    transforms.Resize((480, 640)),  # Resize to webcam frame size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

style_image = load_image(style_image_path, transform)

# Initialize the style transfer model
style_transfer_model = StyleTransferModel(style_image).to(device)
optimizer = torch.optim.Adam([style_image.requires_grad_()], lr=0.003)
mse_loss = nn.MSELoss()

# Function to stylize an image
def stylize_frame(content_img, alpha=0.6):
    # Convert NumPy array to PIL Image
    content_img_pil = Image.fromarray(content_img)
    
    content_img_tensor = transform(content_img_pil).unsqueeze(0).to(device)
    content_img_tensor = content_img_tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([content_img_tensor], lr=0.003)

    for _ in range(10):
        optimizer.zero_grad()
        style_features = style_transfer_model(content_img_tensor)
        style_loss = mse_loss(style_features[0], style_transfer_model.style_features[0])
        style_loss.backward()
        optimizer.step()

    output_img = tensor_to_image(content_img_tensor)
    
    # Ensure both images have the same dimensions
    content_img_resized = cv2.resize(content_img, (output_img.shape[1], output_img.shape[0]))
    
    # Blend the original content image and the stylized image
    blended_img = cv2.addWeighted(content_img_resized, 1 - alpha, output_img, alpha, 0)
    
    return blended_img

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stylized_frame = stylize_frame(frame_rgb, alpha=0.6)
    
    # Convert RGB to BGR for OpenCV
    stylized_frame_bgr = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Stylized Webcam', stylized_frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

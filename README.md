## Real-Time-Style-Transfer

This project implements a real-time style transfer application using a pre-trained VGG model. It allows you to apply the artistic style of an image to your webcam feed, creating a unique visual experience.

## Dependencies
- Python
- OpenCV
- NumPy
- PyTorch
- torchvision
- Pillow (PIL Fork)

## Run the script:

`python transfer.py`

The application will display your webcam feed with the applied style in real-time.

- Adjust the `alpha` parameter in the `stylize_frame` function within the script to control the blend between the original content and the stylized image (higher alpha for stronger style).

- Press 'q' to quit the application.

## How it Works
The application utilizes a pre-trained VGG model to extract content and style features from both the input image (webcam frame) and the style image. It then optimizes the input image to match the style features while preserving the content. Finally, it blends the stylized image with the original content for a seamless effect.

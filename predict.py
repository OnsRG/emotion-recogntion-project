import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
#from src.modeling.model import EmotionCNN
from src.modeling.resnet_model import EmotionResNet


def predict(image_paths, cfg, class_names):
    # Step 1: Load the saved model
    model = EmotionResNet(num_emotions=cfg["num_emotions"], dropout=cfg["dropout"]).to(cfg["device"])
    checkpoint = torch.load(
        f"{cfg['save_path']}/weights/{cfg['experiment']}.pt",
        map_location=cfg["device"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Step 2: Define transforms (no augmentation)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Step 3: Predict each image
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            # Load and process image
            img     = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            tensor  = transform(img).unsqueeze(0).to(cfg["device"])

            # Get prediction
            output  = model(tensor)
            _, pred = torch.max(output, 1)
            emotion = class_names[pred.item()]

            # Plot
            axes[i].imshow(img)
            axes[i].set_title(f"Predicted: {emotion}", fontsize=10)
            axes[i].axis("off")

    plt.suptitle("Emotion Predictions on New Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{cfg['save_path']}/plots/predictions_ResNet.png")
    plt.show()


image_paths = [
    "test_images/angry-face.png",
    "test_images/contempt-face.png",
    "test_images/disgust-face.jpg",
    "test_images/face-fear.jpg",
    "test_images/happy-face.png",
    "test_images/neutral-face.png",
    "test_images/sad-face.png",
    "test_images/surprise-face.jpg",
    "test_images/neutral-face1.jpg",
    "test_images/happy-face1.png",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {
    "num_emotions": 8,
    "dropout"     : 0.6,
    "device"      : device,
    "save_path"   : "outputs",
    "experiment"  : "emotion_classifier_run_4",
    "img_size"    : 224,  # ResNet native size
}

class_names = ['Anger','Contempt','Disgust','Fear',
               'Happy','Neutral','Sad','Surprise']

predict(image_paths, CFG, class_names)    
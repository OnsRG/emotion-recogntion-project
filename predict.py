import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from src.modeling.model import EmotionCNN


def predict(image_paths, cfg, class_names):
    # Step 1: Load the saved model
    model = EmotionCNN(num_emotions=cfg["num_emotions"], dropout=cfg["dropout"]).to(cfg["device"])
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
    plt.savefig(f"{cfg['save_path']}/plots/predictions.png")
    plt.show()
from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)) -> Tuple[str, List[float]]:
    try:
        # 1. Open image
        img = image

        # 2. Create transformation for image
        image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        
        # 3. Move model to the appropriate device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        # 4. Transform the image and ensure it's in float32
        transformed_image = image_transform(img).unsqueeze(dim=0).to(device).float()  # Convert input to float32

        # 5. Make a prediction
        with torch.no_grad():
            target_image_pred = model(transformed_image)

        # 6. Calculate probabilities using softmax
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        st.write("Probabilities after softmax:", target_image_pred_probs.cpu().numpy())

        # 7. Get the predicted class label index
        target_image_pred_label_idx = torch.argmax(target_image_pred_probs, dim=1).item()

        # 8. Get the class label and probabilities
        predicted_class = class_names[target_image_pred_label_idx]
        probabilities = target_image_pred_probs.cpu().numpy().flatten().tolist()

        return predicted_class, probabilities

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", [0.0] * len(class_names)

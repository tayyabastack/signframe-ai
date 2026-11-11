
import matplotlib.pyplot as plt

from src.dataset import SignLanguageDataset, VALID_LABELS


# Load our prepared dataset
dataset = SignLanguageDataset(
    "data/raw/sign_mnist_train.csv"
)

# Create a grid of 12 sample images
fig, axes = plt.subplots(3, 4, figsize=(10, 8))

for i, ax in enumerate(axes.flat):
    image, label = dataset[i]

    # Convert tensor from (1, 28, 28) to (28, 28)
    image = image.squeeze(0).numpy()

    # Convert class index back to ASL letter
    original_label = VALID_LABELS[label.item()]
    letter = chr(ord("A") + original_label)

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Letter: {letter}")
    ax.axis("off")

plt.suptitle("SignFrame AI — Training Dataset Samples")
plt.tight_layout()
plt.show()

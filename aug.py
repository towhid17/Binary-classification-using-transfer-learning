from torchvision import transforms
from PIL import Image

# Define the data augmentation transformations
data_transforms = transforms.Compose([
    # transforms.RandomRotation(degrees=90),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.8),
    # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomNoise(),
    # transforms.ToTensor(),
])

# Load the original 50 bad images
bad_images = [Image.open(f'archive/train/bad/image ({i}).png') for i in range(1, 51)]

# Apply the data augmentation transformations to create 200 new bad images
augmented_images = []
for i in range(200):
    image = bad_images[i % 50]  # Use a different original image for each new image
    augmented_image = data_transforms(image)
    augmented_images.append(augmented_image)

# Save the new bad images to disk
for i, image in enumerate(augmented_images):
    image.save(f'archive/train/aug-bad/image ({i+51}).png')

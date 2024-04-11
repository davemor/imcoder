from PIL import Image
from imcoder.encoders.models import get_model

IMAGE_PATH = '/scratch/cervical_processed/patches_256_1/40/malignant/00003269-IC-CX-00051-01-7424-9216-1-256.png'

model, preprocess = get_model('dino2s')

with Image.open(IMAGE_PATH) as img:
    img = preprocess(img).unsqueeze(0)
    embeddings = model(img)
    embeddings = embeddings.squeeze()
    embeddings = embeddings.detach().numpy()
    print(embeddings.shape)
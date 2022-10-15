from torch.utils.data import DataLoader
from utils import ImageDatasetCV, CycleGAN, save_model


MONET_DIR = 'data/monet_jpg/'
PHOTO_DIR = 'data/photo_jpg/'
SAVE_PATH = 'model'

IMAGE_SIZE = [256, 256]
BATCH_SIZE = 1


if __name__ == '__main__':

    img_dataset = ImageDatasetCV(monet_dir=MONET_DIR, photo_dir=PHOTO_DIR)

    img_dataloader = DataLoader(img_dataset,
                                batch_size=BATCH_SIZE,
                                pin_memory=True)

    gan = CycleGAN(epochs=100)

    gan.train(img_dataloader)

    save_model(gan, SAVE_PATH+'/model')
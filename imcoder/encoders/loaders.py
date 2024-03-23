from pathlib import Path
from typing import Callable, List, Optional, Any
from pprint import pprint

from torch.utils.data import IterableDataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, is_image_file


class UnlabelledImageFolder(VisionDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=None)
        filepaths = [
            p for p in Path(root).glob("./*") if is_image_file(p.as_posix())
        ]
        self.filepaths = sorted(filepaths, key=lambda p: p.stem)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Any:
        path = str(self.filepaths[index])
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SimpleImagesDataset(IterableDataset):
    def __init__(self, paths, transforms, transform, target_transform) -> None:
        self.paths = paths
        self.transforms = transforms

        def __len__(self) -> int:
            return len(paths)
        
    def __getitem__(self, index: int) -> Any:
        sample = default_loader(self.paths[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

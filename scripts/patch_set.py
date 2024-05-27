"""
x, y, label, setting_idx

and a json object:

```javascript
{
    settings: [
        {
            slide:
            annotation:
            category:
            subcategory:
            split:
            level:
            patch_size:
        }
    ]
}
"""


import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
# import cv2

# from histoquery.data.slides.slide import Region



def invert(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


class PatchSetting:
    def __init__(self, slide, annotation, category, subcategory, split, level, patch_size) -> None:
        self.slide = slide
        self.annotation = annotation
        self.category = category
        self.subcategory = subcategory
        self.split = split
        self.level = level
        self.patch_size = patch_size

    def to_dict(self):
        d = self.__dict__
        d['slide'] = str(self.slide)
        d['annotation'] = str(self.annotation)
        return d
    
    @classmethod
    def from_dict(cls, d):
        d['slide'] = Path(d['slide'])
        d['annotation'] = Path(d['annotation'])
        return cls(**d)


class PatchSet:
    def __init__(self, df: pd.DataFrame, settings: List[PatchSetting]) -> None:
        """The dataframe should have the following columns:
            - x: left position of the patch at level
            - y: top position of the patch at level
            - label: which class it belongs to
            - setting: an index into the settings array.

        Args:
            df (pd.DataFrame): The patch locations, labels, and index into settings.
            settings (List[PatchSetting]): A list of settings.
        """
        self.df = df
        self.settings = settings

    def save(self, path: Path) -> None:
        """Saves a PatchSet to disk

        The dataframe is saved to a csv called frame.csv
        The settings are saved in a text file called settings.json

        Args:
            path (Path): the directory in which to save the patchset
        """
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / "frame.csv", index=False)
        dicts = [s.to_dict() for s in self.settings]
        with open(path / "settings.json", "w") as outfile:
            json.dump(dicts, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        """Loads a PatchSet from disk

        Assumes:
        The dataframe is saved to a csv called frame.csv
        The settings are saved in a text file called settings.json

        Args:
            path (Path): the directory in which the patchset is saved
        """
        df = pd.read_csv(path / "frame.csv")
        with open(path / "settings.json") as json_file:
            settings = json.load(json_file)
            settings = [PatchSetting.from_dict(s) for s in settings]
        return cls(df, settings)
    

    def description(self):
        """Returns basic summary of patchset

        returns the labels and the total number of patches of each label

        """
        labels = np.unique(self.df.label)
        sum_totals = [np.sum(self.df.label == label) for label in labels]
        return labels, sum_totals
    

    # def export_patches(self, output_dir, slide_cls, labels) -> None:
    #     """Creates all patches in a patch set

    #     Writes patches in subdirectories of their label
    #     Patches are name slide_path_x_y_level_patch_size.png

    #     Args:
    #         output_dir (Path): the directory in which the patches are saved
    #     """
    #     groups = self.df.groupby("setting")
    #     for setting_idx, group in groups:
    #         s = self.settings[setting_idx]
    #         self._export_patches_for_setting(
    #             group, output_dir, s.slide, s.level, s.patch_size, slide_cls, labels
    #         )

    # def _export_patches_for_setting(
    #     self,
    #     frame: pd.DataFrame,
    #     output_dir: Path,
    #     slide_path: Path,
    #     level: int,
    #     patch_size: int,
    #     slide_cls,
    #     labels,
    # ):
    #     """Creates all the patches for an individual PatchSetting"""

    #     def get_output_dir_for_label(label: str) -> Path:
    #         label_str = invert(labels)[label]
    #         label_dir = Path(output_dir) / label_str
    #         return label_dir

    #     def make_patch_path(idx: int, x: int, y: int, label: int) -> Path:
    #         filename = f"{idx:08}-{Path(slide_path).stem}-{x}-{y}-{level}-{patch_size}.png"
    #         label_dir = get_output_dir_for_label(label)
    #         label_dir.mkdir(parents=True, exist_ok=True)
    #         return label_dir / filename

    #     def save_patch(region: Region, slide, filepath: Path) -> None:
    #         image = slide.read_region(region)
    #         opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(str(filepath), np.array(opencv_image))

    #     with slide_cls(slide_path) as slide:
    #         for row in frame.itertuples():
    #             filepath = make_patch_path(row.Index, row.x, row.y, row.label)
    #             if not filepath.exists():  # only write the file if it doesn't exist
    #                 region = Region.make(row.x, row.y, patch_size, patch_size, level)
    #                 save_patch(region, slide, filepath)
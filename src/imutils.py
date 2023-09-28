import json
import numpy as np
from pathlib import Path
from typing import Dict
from skimage import io, measure, morphology, feature


def generate_selections(image_folder: Path,
                        annotations_folder: Path,
                        out_folder: Path):
    """
    Crop OCTA images according to the selection given in the RH mask.
    More precisely, it crops the images creating a box which is four times (i.e. each side is doubled) the bounding
    box of the annotated RH.
    """
    # get patients RH masks
    rh_masks_list = list(annotations_folder.glob("*_RH_mask.png"))

    for rh_mask in rh_masks_list:
        # read rh mask
        rh_mask_arr = io.imread(rh_mask, as_gray=True).astype(bool)
        # get region and check if there is only one region
        regions = measure.regionprops(measure.label(rh_mask_arr))
        assert len(regions) == 1, "The should be only one annotated RH in the image"
        # get region bounding box
        region_props = regions[0]
        min_i, min_j, max_i, max_j = region_props.bbox
        length_i = max_i - min_i
        length_j = max_j - min_j
        # generate selection as a bounding box with double sides
        selection_min_i = min_i - int(np.round(length_i / 2))
        selection_max_i = max_i + int(np.round(length_i / 2))
        selection_min_j = min_j - int(np.round(length_j / 2))
        selection_max_j = max_j + int(np.round(length_j / 2))

        # get patient
        patient = rh_mask.stem.split('_')[0]
        # select OCTA images
        octas_list_for_patient = list(image_folder.glob(f"{patient}_OCTA_*"))
        # crop each image
        for octa_im in octas_list_for_patient:
            octa_im_arr = io.imread(octa_im)  # read
            octa_crop = octa_im_arr[selection_min_i:selection_max_i, selection_min_j:selection_max_j]  # crop
            io.imsave(out_folder / Path(f"{octa_im.stem}_crop.png"), octa_crop)


def generate_edges_and_skeleton_for_pic2d(pic2d: Path):
    """
    Save the edges and the skeleton of the given Putative Initial Condition binary
    """
    patient = pic2d.stem.split("_")[0]
    pic2d_arr = io.imread(pic2d, as_gray=True).astype(bool)
    skeleton = morphology.skeletonize(pic2d_arr)
    io.imsave(pic2d.parent / Path(f"{patient}_pic2d_skeleton.png"), skeleton)
    edges = morphology.skeletonize(feature.canny(pic2d_arr, sigma=0.8))
    io.imsave(pic2d.parent / Path(f"{patient}_pic2d_edges.png"), edges)


if __name__ == "__main__":
    for pic2d in Path("../input_patients_data/3_pic_binaries").iterdir():
        generate_edges_and_skeleton_for_pic2d(pic2d)

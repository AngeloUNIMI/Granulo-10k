import os
import math
import yaml  # make sure pyyaml is installed: pip install pyyaml

import torch
from torchvision import datasets
from torchvision.transforms import functional as F


def load_stereo_calib_yaml(calib_path, device="cpu", dtype=torch.float32):
    """Load a simple OpenCV-style stereo calibration YAML.

    Expected keys in the YAML:
        K1, D1, K2, D2, R, T, image_size, E, F

    Only K1, K2, R, T are required here.
    """
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)

    def to_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device)

    K1 = to_tensor(data["K1"])
    K2 = to_tensor(data["K2"])
    R = to_tensor(data["R"])
    T = to_tensor(data["T"]).view(3, 1)

    # Camera 1 is the world/reference camera
    R1 = torch.eye(3, dtype=dtype, device=device)
    t1 = torch.zeros(3, 1, dtype=dtype, device=device)

    R2 = R
    t2 = T

    # 3x4 projection matrices P = K [R | t]
    P1 = K1 @ torch.cat([R1, t1], dim=1)
    P2 = K2 @ torch.cat([R2, t2], dim=1)

    return {
        "K1": K1,
        "K2": K2,
        "R1": R1,
        "t1": t1,
        "R2": R2,
        "t2": t2,
        "P1": P1,
        "P2": P2,
        "raw": data,
    }


def hflip_homography(width: int, height: int, device=None, dtype=torch.float32):
    """Homography for a horizontal flip.

    Maps (u_out, v_out) in the *flipped* image to (u_in, v_in) in the original.
    """
    if device is None:
        device = torch.device("cpu")
    return torch.tensor(
        [
            [-1.0, 0.0, width - 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def rotation_center_homography(width: int, height: int, angle_deg: float,
                               device=None, dtype=torch.float32):
    """Homography for rotation by angle_deg (degrees) around image center.

    angle_deg: same convention as torchvision.transforms.functional.rotate,
               positive is counter-clockwise.
    Maps (u_out, v_out) in the *rotated* image to (u_in, v_in) in the original.
    """
    if device is None:
        device = torch.device("cpu")
    theta = math.radians(angle_deg)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    # translation to center
    T1 = torch.tensor(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    # rotation (output->input uses -theta, which gives this matrix form)
    R = torch.tensor(
        [
            [math.cos(theta), math.sin(theta), 0.0],
            [-math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    # translation back
    T2 = torch.tensor(
        [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return T2 @ R @ T1


class ImageFolderWithSizeDA(datasets.ImageFolder):
    """Custom dataset that loads two synchronized views + extra metadata.

    This version also:
      * loads stereo calibration from an OpenCV-style YAML file
      * can apply a simple two-view augmentation (independent view-level
        augmentation for cam A and cam B) and updates the per-view projection
        matrices accordingly.

    If two_view_augmentation=False, behaviour is identical to your original
    version: the same transform is applied to both images via
    `self.transform(imA, imB)`.
    """

    def __init__(
        self,
        dirDbOrig,
        image_pathP,
        transformP,
        misure,
        misureNorm,
        panNumbers,
        infoSpessore,
        calib_path=None,
        two_view_augmentation=False,
        max_rotation_deg=5.0,
        device="cpu",
    ):
        super(ImageFolderWithSizeDA, self).__init__(root=image_pathP, transform=transformP)
        # You weren't really using self.data, but we keep it in case it's used elsewhere
        self.data = datasets.ImageFolder(image_pathP, transformP)

        self.misure = misure
        self.misureNorm = misureNorm
        self.panNumbers = panNumbers
        self.infoSpessore = infoSpessore

        self.dirFilesB = os.path.join(dirDbOrig, "datastore_B")

        # Calibration / augmentation config
        self.two_view_augmentation = two_view_augmentation
        self.max_rotation_deg = max_rotation_deg
        self.device = torch.device(device)

        if calib_path is not None:
            self.calib = load_stereo_calib_yaml(calib_path, device=self.device)
        else:
            self.calib = None

        if self.two_view_augmentation and self.calib is None:
            raise ValueError(
                "two_view_augmentation=True but no calib_path was provided. "
                "Please pass calib_path pointing to your calib_opencv_simple.yml."
            )

    def _augment_single_view(self, img, width, height):
        """Apply random view-level aug (rotation + optional h-flip) to one PIL image.

        Returns:
            img_aug (PIL.Image)
            H      (3x3 tensor): homography mapping augmented coords -> original coords
        """
        # Start with identity homography
        H_total = torch.eye(3, dtype=torch.float32, device=self.device)
        img_aug = img

        # Random small rotation around image center
        angle = float(torch.empty(1).uniform_(-self.max_rotation_deg, self.max_rotation_deg))
        if abs(angle) > 1e-3:
            H_rot = rotation_center_homography(width, height, angle, device=self.device)
            img_aug = F.rotate(img_aug, angle=angle, expand=False)
            # compose: old -> new op on the right
            H_total = H_total @ H_rot

        # Random horizontal flip with p=0.5
        if bool(torch.randint(0, 2, (1,))):
            H_flip = hflip_homography(width, height, device=self.device)
            img_aug = F.hflip(img_aug)
            H_total = H_total @ H_flip

        return img_aug, H_total

    def _apply_two_view_augmentation(self, imA, imB):
        """Apply independent view-level augmentation to cam A and cam B.

        Returns:
            imA_aug, imB_aug: augmented PIL images
            P1_aug, P2_aug:  3x4 projection matrices after augmentation
        """
        # Use the actual image size (in case it's different from what is in the YAML)
        widthA, heightA = imA.size
        widthB, heightB = imB.size

        # --- View A ---
        imA_aug, H1 = self._augment_single_view(imA, widthA, heightA)
        # --- View B ---
        imB_aug, H2 = self._augment_single_view(imB, widthB, heightB)

        # Update projection matrices: P' = H^{-1} P
        P1 = self.calib["P1"]
        P2 = self.calib["P2"]

        H1_inv = torch.inverse(H1)
        H2_inv = torch.inverse(H2)

        P1_aug = H1_inv @ P1
        P2_aug = H2_inv @ P2

        return imA_aug, imB_aug, P1_aug, P2_aug

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # the image file path (view A)
        path = self.imgs[index][0]
        imA = self.loader(path)  # first view (A)

        dir_name, filename = os.path.split(path)
        C = filename.split("_")
        id_pan = int(C[0])
        indexL = self.panNumbers.index(id_pan)

        classV = self.misure[indexL]
        classV_norm = self.misureNorm[indexL]

        # check for spessore
        rootFileName = C[0] + "_" + C[1]

        try:
            _ = self.infoSpessore.index(rootFileName)
            # height visible, width not visible, thickness visible
            weightVector = torch.tensor([1, 0, 1], dtype=torch.float32)
        except ValueError:
            # height visible, width visible, thickness not visible
            weightVector = torch.tensor([1, 1, 0], dtype=torch.float32)

        # second view (B) is stored in a parallel directory structure
        baseDir, dirPanel = os.path.split(dir_name)
        fileB = os.path.join(self.dirFilesB, dirPanel, rootFileName + "_B.jpg")

        imB = self.loader(fileB)

        # --- Two-view augmentation (optional) ---
        cam_params = None
        if self.two_view_augmentation and self.calib is not None:
            imA_aug, imB_aug, P1_aug, P2_aug = self._apply_two_view_augmentation(imA, imB)

            # If you still want to apply your original transformP (e.g. color jitter,
            # normalization), keep it *photometric-only* when using two-view augmentation,
            # so that geometry stays consistent with P1_aug / P2_aug.
            if self.transform is not None:
                imA_aug, imB_aug = self.transform(imA_aug, imB_aug)

            cam_params = {
                "P1": P1_aug,
                "P2": P2_aug,
                "K1": self.calib["K1"],
                "K2": self.calib["K2"],
                "R1": self.calib["R1"],
                "t1": self.calib["t1"],
                "R2": self.calib["R2"],
                "t2": self.calib["t2"],
            }

        else:
            # Original behaviour: same transform on two images
            if self.transform is not None:
                imA_aug, imB_aug = self.transform(imA, imB)
            else:
                imA_aug, imB_aug = imA, imB

        # convert labels to tensors
        classV_tensor = torch.tensor(classV)
        classV_norm_tensor = torch.tensor(classV_norm)

        # Return camera parameters only when augmentation is enabled.
        if cam_params is not None:
            return (
                imA_aug,
                imB_aug,
                id_pan - 1,
                path,
                0, # mask A
                0, # mask B               
                classV_tensor,
                classV_norm_tensor,
                weightVector,
                # cam_params,
            )
        else:
            return (
                imA_aug,
                imB_aug,
                id_pan - 1,
                path,
                0, # mask A
                0, # mask B
                classV_tensor,
                classV_norm_tensor,
                weightVector,
            )

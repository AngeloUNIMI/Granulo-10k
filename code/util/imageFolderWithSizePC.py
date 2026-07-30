import os
import math
import yaml
import glob
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import functional as F
import warnings

def load_stereo_calib_yaml(calib_path, device="cpu", dtype=torch.float32):
    """Load a simple OpenCV-style stereo calibration YAML."""
    with open(calib_path, "r") as f:
        data = yaml.safe_load(f)

    def to_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device)

    K1 = to_tensor(data["K1"])
    K2 = to_tensor(data["K2"])
    R = to_tensor(data["R"])
    T = to_tensor(data["T"]).view(3, 1)

    R1 = torch.eye(3, dtype=dtype, device=device)
    t1 = torch.zeros(3, 1, dtype=dtype, device=device)

    R2 = R
    t2 = T

    P1 = K1 @ torch.cat([R1, t1], dim=1)
    P2 = K2 @ torch.cat([R2, t2], dim=1)

    return {
        "K1": K1, "K2": K2, "R1": R1, "t1": t1,
        "R2": R2, "t2": t2, "P1": P1, "P2": P2, "raw": data,
    }


def hflip_homography(width: int, height: int, device=None, dtype=torch.float32):
    if device is None: device = torch.device("cpu")
    return torch.tensor([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)


def rotation_center_homography(width: int, height: int, angle_deg: float, device=None, dtype=torch.float32):
    if device is None: device = torch.device("cpu")
    theta = math.radians(angle_deg)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    T1 = torch.tensor([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    R = torch.tensor(
        [[math.cos(theta), math.sin(theta), 0.0], [-math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype, device=device)
    T2 = torch.tensor([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    return T2 @ R @ T1


class ImageFolderWithSizePC(datasets.ImageFolder):
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
            sample_points=2048,
            augment_pc=False,
    ):
        super(ImageFolderWithSizePC, self).__init__(root=image_pathP, transform=transformP)
        self.data = datasets.ImageFolder(image_pathP, transformP)

        self.misure = misure
        self.misureNorm = misureNorm
        self.panNumbers = panNumbers
        self.infoSpessore = infoSpessore
        self.augment_pc = augment_pc

        # Paths for secondary data
        self.dirFilesB = os.path.join(dirDbOrig, "datastore_B")
        self.dirFilesPC = os.path.join(dirDbOrig, "datastore_PC")

        self.sample_points = sample_points

        # Calibration / augmentation config
        self.two_view_augmentation = two_view_augmentation
        self.max_rotation_deg = max_rotation_deg
        self.device = torch.device(device)

        if calib_path is not None:
            self.calib = load_stereo_calib_yaml(calib_path, device=self.device)
        else:
            self.calib = None

        if self.two_view_augmentation and self.calib is None:
            raise ValueError("two_view_augmentation=True but no calib_path provided.")

    def _augment_single_view(self, img, width, height):
        H_total = torch.eye(3, dtype=torch.float32, device=self.device)
        img_aug = img
        angle = float(torch.empty(1).uniform_(-self.max_rotation_deg, self.max_rotation_deg))
        if abs(angle) > 1e-3:
            H_rot = rotation_center_homography(width, height, angle, device=self.device)
            img_aug = F.rotate(img_aug, angle=angle, expand=False)
            H_total = H_total @ H_rot
        if bool(torch.randint(0, 2, (1,))):
            H_flip = hflip_homography(width, height, device=self.device)
            img_aug = F.hflip(img_aug)
            H_total = H_total @ H_flip
        return img_aug, H_total

    def _apply_two_view_augmentation(self, imA, imB):
        widthA, heightA = imA.size
        widthB, heightB = imB.size
        imA_aug, H1 = self._augment_single_view(imA, widthA, heightA)
        imB_aug, H2 = self._augment_single_view(imB, widthB, heightB)
        P1 = self.calib["P1"]
        P2 = self.calib["P2"]
        H1_inv = torch.inverse(H1)
        H2_inv = torch.inverse(H2)
        P1_aug = H1_inv @ P1
        P2_aug = H2_inv @ P2
        return imA_aug, imB_aug, P1_aug, P2_aug

    def _resample(self, points, k):
        """
        Resample point cloud to have exactly k points.
        If 'points' is empty (no file or empty file), returns a zero-tensor.
        """
        # 1. Handle Empty Case
        # checks if array is empty or None
        if points is None or points.size == 0:
            return np.zeros((k, 3), dtype=np.float32)

        # 2. Resample
        if len(points) >= k:
            # Downsample: Choose k unique indices
            choice_idx = np.random.choice(len(points), k, replace=False)
        else:
            # Upsample: Sample with replacement to reach k
            choice_idx = np.random.choice(len(points), k, replace=True)

        return points[choice_idx]

    def _load_point_cloud(self, dir_panel, root_filename):
        suffixes = ["_PC_lungh_largh.xyz", "_PC_thickness.xyz"]

        for suffix in suffixes:
            path_candidate = os.path.join(self.dirFilesPC, dir_panel, root_filename + suffix)

            if not os.path.exists(path_candidate):
                continue

            # Check for 0-byte files
            if os.path.getsize(path_candidate) == 0:
                continue

            try:
                # Context manager to suppress 'UserWarning: loadtxt: input contained no data'
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*loadtxt: input contained no data.*")
                    loaded = np.loadtxt(path_candidate, dtype=np.float32, ndmin=2)

                if loaded.size == 0:
                    continue

                if loaded.shape[1] != 3:
                    if loaded.size % 3 == 0:
                        loaded = loaded.reshape(-1, 3)
                    else:
                        continue

                return loaded

            except Exception:
                continue

        return np.zeros((0, 3), dtype=np.float32)

    def _augment_point_cloud(self, pc_np):
        """
        Applies random rotation about Z-axis and random jitter.
        Args:
            pc_np: numpy array of shape (N, 3)
        """
        # 1. Random Rotation around Z-axis
        # We rotate only X and Y, keeping Z fixed (gravity direction)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Apply rotation: (N, 3) dot (3, 3) -> (N, 3)
        pc_np = np.dot(pc_np, rotation_matrix.T)

        # 2. Random Jitter (Gaussian Noise)
        # sigma=0.01 means 1cm noise if units are meters, adjust based on your units!
        sigma = 0.005
        clip = 0.02
        
        # jitter ONLY in XY
        jitter_xy = np.clip(
            sigma * np.random.randn(pc_np.shape[0], 2),
            -clip, clip
        )
        pc_np[:, :2] += jitter_xy

        return pc_np.astype(np.float32)

    def _normalize_point_cloud(self, pc_np):
        """
        Center point cloud
        """
        if pc_np.shape[0] == 0:
            return pc_np.astype(np.float32)

        centroid = np.mean(pc_np, axis=0, keepdims=True)
        pc_np = pc_np - centroid

        return pc_np.astype(np.float32)

    def _compute_dimensions(self, pc_np):
        """
        Compute [width, height, thickness] from point cloud.
        """
        if pc_np.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        mins = pc_np.min(axis=0)
        maxs = pc_np.max(axis=0)
        dims = maxs - mins   # [X, Y, Z]

        return dims.astype(np.float32)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        imA = self.loader(path)

        dir_name, filename = os.path.split(path)
        C = filename.split("_")
        id_pan = int(C[0])
        indexL = self.panNumbers.index(id_pan)

        classV = self.misure[indexL]
        classV_norm = self.misureNorm[indexL]
        rootFileName = C[0] + "_" + C[1]

        # Determine visibility
        try:
            _ = self.infoSpessore.index(rootFileName)
            weightVector = torch.tensor([1, 0, 1], dtype=torch.float32)
        except ValueError:
            weightVector = torch.tensor([1, 1, 0], dtype=torch.float32)

        # Load View B
        baseDir, dirPanel = os.path.split(dir_name)
        fileB = os.path.join(self.dirFilesB, dirPanel, rootFileName + "_B.jpg")
        imB = self.loader(fileB)

        # --- Load Point Cloud ---
        pc_raw = self._load_point_cloud(dirPanel, rootFileName)

        # Resample
        pc_processed = self._resample(pc_raw, self.sample_points)

        # Augment (optional)
        if self.augment_pc:
            pc_processed = self._augment_point_cloud(pc_processed)

        # NORMALIZE (CRITICAL)
        pc_processed = self._normalize_point_cloud(pc_processed)

        # Compute regression target
        dims = self._compute_dimensions(pc_processed)  # [W, H, T]

        pc_tensor = torch.from_numpy(pc_processed).float()
        dims_tensor = torch.from_numpy(dims).float()

        # Augmentation
        cam_params = None
        if self.two_view_augmentation and self.calib is not None:
            imA_aug, imB_aug, P1_aug, P2_aug = self._apply_two_view_augmentation(imA, imB)
            if self.transform is not None:
                imA_aug, imB_aug = self.transform(imA_aug, imB_aug)
            cam_params = {
                "P1": P1_aug, "P2": P2_aug,
                "K1": self.calib["K1"], "K2": self.calib["K2"],
                "R1": self.calib["R1"], "t1": self.calib["t1"],
                "R2": self.calib["R2"], "t2": self.calib["t2"],
            }
        else:
            if self.transform is not None:
                imA_aug, imB_aug = self.transform(imA, imB)
            else:
                imA_aug, imB_aug = imA, imB

        classV_tensor = torch.tensor(classV)
        classV_norm_tensor = torch.tensor(classV_norm)

        if cam_params is not None:
            return (
                imA_aug, imB_aug, pc_tensor, dims_tensor,
                0, # mask A
                0, # mask B
                id_pan - 1, path,
                classV_tensor, classV_norm_tensor, weightVector,
                # cam_params,
            )
        else:
            return (
                imA_aug, imB_aug, pc_tensor, dims_tensor,
                0, # mask A
                0, # mask B
                id_pan - 1, path,
                classV_tensor, classV_norm_tensor, weightVector,
            )
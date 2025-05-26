import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from utils.resample_3d_curve import resample_3d_curve


@dataclass
class LaDEEPDataLoader(Dataset):
    strip_path: str
    mould_path: str
    section_path: str
    params_path: str
    load_path: str
    unload_path: str
    mode: str = "train"
    n_type: int = 5
    n_points: float = 301
    numeric_type = np.float32

    def __post_init__(self):

        self.strip_line_paths = self._read_paths(self.strip_path)
        self.mould_line_paths = self._read_paths(self.mould_path)
        self.section_sdf_paths = self._read_paths(self.section_path)
        self.params_paths = self._read_paths(self.params_path)
        self.load_line_paths = self._read_paths(self.load_path)
        self.unload_line_paths = self._read_paths(self.unload_path)

        self.scale_factor_for_mould = [1.6669, 0.4628, -0.0002, 1.5532, 0.0002, -0.4709]
        self.scale_factor_for_unload = [1.8492, 0.4575, 0.0835, 1.6107, -0.0007, -0.4770]
        self.scale_factor_for_load = [1.8536, 0.4950, 0.0047, 1.5931, -0.0025, -0.5031]

    def _data_split(self, n):
        if self.mode == "train":
            start, end = 0, int(n // 10 * 8)
        elif self.mode == "eval":
            start, end = int(n // 10 * 8), int(n // 10 * 9)
        elif self.mode == "test":
            start, end = int(n // 10 * 9), n
        else:
            raise "Mode type error!"
        return start, end

    @staticmethod
    def read_2d_array_from_txt(line_path):
        with open(line_path, 'r', encoding = "utf8") as f:
            points = np.array(
                list(
                    map(
                        lambda x: list(
                            map(
                                lambda y: float(y),
                                x.split()
                            )
                        ),
                        f.readlines()
                    )
                )
            )
        return points

    def _read_paths(self, path):
        paths = []
        for i in range(self.n_type):
            type_path = os.path.join(path, f"type_{i}")
            files = os.listdir(type_path)
            start, end = self._data_split(len(files))
            files.sort()
            files = list(map(lambda x: os.path.join(type_path, x), files))
            paths += files[start: end]

        return paths

    def _read_line_diff(self, path, scale_factor_type = "unloading"):
        line = self.read_2d_array_from_txt(path)
        if len(line) != self.n_points:
            line = resample_3d_curve(line, self.n_points)
        line_diff = np.diff(line, axis = 0)
        if scale_factor_type == "unloading":
            scale_factor = self.scale_factor_for_unload
        elif scale_factor_type == "loading":
            scale_factor = self.scale_factor_for_load
        else:
            scale_factor = self.scale_factor_for_mould

        for i in range(3):
            line_diff[:, i] = (line_diff[:, i] - scale_factor[i + 3]) / (scale_factor[i] - scale_factor[i + 3])

        return line_diff

    def _read_section_sdf(self, path):
        return np.expand_dims(cv2.imread(path, cv2.IMREAD_UNCHANGED), axis = 0)

    def _read_section_sdf_deprecated(self, path):
        sdf = self.read_2d_array_from_txt(path)
        min_val, max_val = np.min(sdf), np.max(sdf)
        normalized_sdf = (sdf - min_val) / (max_val - min_val)
        normalized_sdf = np.expand_dims(normalized_sdf, axis = 0)
        return normalized_sdf

    def _read_params(self, path):
        params = np.loadtxt(path, delimiter = ",")
        params = np.expand_dims(np.sum(params, axis = 0), axis = 0)
        min_val, max_val = np.min(params[0, :3]), np.max(params[0, :3])
        params[0, :3] = (params[0, :3] - min_val) / (max_val - min_val)
        min_val, max_val = np.min(params[0, 3:]), np.max(params[0, 3:])
        params[0, 3:] = (params[0, 3:] - min_val) / (max_val - min_val)
        return params

    def __getitem__(self, index):
        strip_line_diff = self._read_line_diff(
            self.strip_line_paths[index]
        ).T
        mould_line_diff = self._read_line_diff(
            self.mould_line_paths[index],
            scale_factor_type = "mould"
        ).T
        load_line_diff = self._read_line_diff(
            self.load_line_paths[index],
            scale_factor_type = "loading"
        ).T
        unload_line_diff = self._read_line_diff(
            self.unload_line_paths[index]
        ).T

        section_sdf = self._read_section_sdf(
            self.section_sdf_paths[index]
        )
        params = self._read_params(
            self.params_paths[index]
        )

        return \
            list(
                map(
                    lambda x: x.astype(self.numeric_type),
                    (
                        strip_line_diff,
                        mould_line_diff,
                        section_sdf,
                        params,
                        load_line_diff,
                        unload_line_diff
                    )
                )
            )

    def __len__(self):
        return len(self.strip_line_paths)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import os
from os.path import join as pjoin
import subprocess
import sys

import numpy as np


class COLMAPParams:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("image_path", help="image path")
        self.parser.add_argument("workspace_path", help="workspace path")
        self.parser.add_argument(
            "--mask_path",
            help="path for mask to exclude feature extration from those regions",
            default=None,
        )
        self.parser.add_argument(
            "--dense_max_size", type=int, help='Max size for dense COLMAP', default=384,
        )
        self.add_arguments(self.parser)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--colmap_bin_path",
            help="path to colmap bin. COLMAP 3.6 is required to enable mask_path",
            default='colmap'
        )
        parser.add_argument(
            "--sparse", help="disable dense reconstruction", action='store_true'
        )
        parser.add_argument(
            "--initialize_pose", help="Intialize Pose", action='store_true'
        )
        parser.add_argument(
            "--camera_params", help="prior camera parameters", default=None
        )
        parser.add_argument(
            "--camera_model", help="camera_model", default='SIMPLE_PINHOLE'
        )
        parser.add_argument(
            "--refine_intrinsics",
            help="refine camera parameters. Not used when camera_params is None",
            action="store_true"
        )
        parser.add_argument(
            "--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
            help="COLMAP matcher ('exhaustive' or 'sequential')"
        )

    def parse_args(self, args=None, namespace=None):
        return self.parser.parse_args(args, namespace=namespace)


class COLMAPProcessor:
    def __init__(self, colmap_bin: str = 'colmap'):
        self.colmap_bin = colmap_bin

    def process(self, args):
        os.makedirs(args.workspace_path, exist_ok=True)

        self.extract_features(args)
        self.match(args)
        if args.initialize_pose:
            self.triangulate(args)
        else:
            self.map(args)

        models = os.listdir(self.sparse_dir(args.workspace_path))
        num_models = len(models)
        logging.info('#models = %d', num_models)
        if num_models > 1:
            logging.error(
                "COLMAP reconstructs more than one model (#models=%d)",
                num_models
            )

        if 'sparse' not in vars(args) or not args.sparse:
            for sub_model in models:
                self.dense(sub_model, args)

    def extract_features(self, args):
        cmd = [
            self.colmap_bin,
            'feature_extractor',
            '--database_path', self.db_path(args.workspace_path),
            '--image_path', args.image_path,
            '--ImageReader.camera_model', args.camera_model,
            '--ImageReader.single_camera', '1'
        ]
        if args.camera_params:
            cmd.extend(['--ImageReader.camera_params', args.camera_params])

        if args.mask_path:
            cmd.extend(['--ImageReader.mask_path', args.mask_path])

        if args.initialize_pose:
            cmd.extend(['--SiftExtraction.num_threads', '1'])
            cmd.extend(['--SiftExtraction.gpu_index', '0'])

        run(cmd)

    def match(self, args):
        cmd = [
            self.colmap_bin,
            f'{args.matcher}_matcher',
            '--database_path', self.db_path(args.workspace_path),
            '--SiftMatching.guided_matching', '1',
        ]
        if args.matcher == "sequential":
            cmd.extend([
                '--SequentialMatching.overlap', '50',
                '--SequentialMatching.quadratic_overlap', '0',
            ])
        run(cmd)

    def triangulate(self, args):
        if self.check_sparse(self.sparse_dir(args.workspace_path, model_index=0)):
            return

        pose_init_dir = self.pose_init_dir(args.workspace_path)
        assert self.check_sparse(pose_init_dir)

        sparse_dir = self.sparse_dir(args.workspace_path, model_index=0)
        os.makedirs(sparse_dir, exist_ok=True)
        cmd = [
            self.colmap_bin,
            'point_triangulator',
            '--database_path', self.db_path(args.workspace_path),
            '--image_path', args.image_path,
            '--output_path', sparse_dir,
            '--input_path', pose_init_dir,
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_local_max_num_iterations', '0',
            '--Mapper.ba_global_max_num_iterations', '1',
        ]
        run(cmd)

    def map(self, args):
        if self.check_sparse(self.sparse_dir(args.workspace_path, model_index=0)):
            return

        sparse_dir = self.sparse_dir(args.workspace_path)
        os.makedirs(sparse_dir, exist_ok=True)
        cmd = [
            self.colmap_bin,
            'mapper',
            '--database_path', self.db_path(args.workspace_path),
            '--image_path', args.image_path,
            '--output_path', sparse_dir,
            # add the following options for KITTI evaluation. Should help in general.
            '--Mapper.abs_pose_min_inlier_ratio', '0.5',
            '--Mapper.abs_pose_min_num_inliers', '50',
            '--Mapper.init_max_forward_motion', '1',
            '--Mapper.ba_local_num_images', '15',
        ]
        if args.camera_params and not args.refine_intrinsics:
            cmd.extend([
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_extra_params', '0',
            ])
        run(cmd)

    def dense(self, recon_model: str, args):
        dense_dir = self.dense_dir(args.workspace_path, model_index=recon_model)
        if self.check_dense(dense_dir, args.image_path):
            return

        os.makedirs(dense_dir, exist_ok=True)

        cmd = [
            self.colmap_bin,
            'image_undistorter',
            '--image_path', args.image_path,
            '--input_path',
            self.sparse_dir(args.workspace_path, model_index=recon_model),
            '--output_path', dense_dir,
            '--output_type', "COLMAP",
            '--max_image_size', str(args.dense_max_size),
        ]
        run(cmd)

        cmd = [
            self.colmap_bin,
            'patch_match_stereo',
            '--workspace_path', dense_dir,
            '--workspace_format', "COLMAP",
            '--PatchMatchStereo.max_image_size', str(args.dense_max_size),
        ]
        run(cmd)

    @staticmethod
    def dense_depth_suffix():
        return ".geometric.bin"

    @staticmethod
    def db_path(workspace):
        return pjoin(workspace, 'database.db')

    @staticmethod
    def sparse_dir(workspace, model_index=None):
        p = pjoin(workspace, 'sparse')
        if model_index is None:
            return p
        return pjoin(p, str(model_index))

    @staticmethod
    def dense_dir(workspace, model_index=None):
        p = pjoin(workspace, 'dense')
        if model_index is None:
            return p
        return pjoin(p, str(model_index))

    @staticmethod
    def pose_init_dir(workspace):
        return pjoin(workspace, 'pose_init')

    @staticmethod
    def check_sparse(sparse_model_dir: str):
        return any(
            all(
                (os.path.isfile(pjoin(sparse_model_dir, name))
                for name in ["cameras" + ext, "images" + ext])
            )
            for ext in ['.bin', '.txt']
        )

    @classmethod
    def check_dense(cls, dense_model_dir: str, image_path: str, valid_ratio=1):
        assert valid_ratio <= 1

        depth_fmt = pjoin(
            dense_model_dir, "stereo", "depth_maps", "{}" + cls.dense_depth_suffix()
        )
        color_names = os.listdir(image_path)

        num_valid = np.sum(os.path.isfile(depth_fmt.format(n)) for n in color_names)
        return (num_valid / len(color_names)) >= valid_ratio


def run(cmd):
    print(' '.join(cmd))
    subprocess.run(cmd)


def main(args):
    processor = COLMAPProcessor(args.colmap_bin)
    processor.process(args)
    return 0


def parse_args():
    return COLMAPParams().parser.parse_args()


if __name__ == '__main__':
    sys.exit(main(parse_args()))

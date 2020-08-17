#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from params import Video3dParamsParser
from process import DatasetProcessor


if __name__ == "__main__":
    parser = Video3dParamsParser()
    params = parser.parse()

    dp = DatasetProcessor()
    dp.process(params)

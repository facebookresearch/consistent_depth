#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def mkdir_ifnotexists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def print_title(text):
    print()
    print("-" * len(text))
    print(text)
    print("-" * len(text))


def print_banner(text):
    w = 12 + len(text)
    print()
    print("*" * w)
    print(f"{'*' * 4}  {text}  {'*' * 4}")
    print("*" * w)


class SuppressedStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

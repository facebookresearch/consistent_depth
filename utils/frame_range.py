#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Set, Optional
from collections import namedtuple

# set is an OptionalSet as below
NamedOptionalSet = namedtuple("NamedOptionalSet", ["name", "set"])


class OptionalSet:
    def __init__(self, set: Optional[Set] = None):
        self.set = set

    def intersection(self, other):
        if self.set is None:
            return other
        if other.set is None:
            return self
        return OptionalSet(set=self.set.intersection(other.set))

    def __str__(self):
        return str(self.set)


class FrameRange:
    """
    Compute the indices of frames we are interested in from the specified range.
    """
    def __init__(
        self,
        frame_range: OptionalSet,
        num_frames: int = None,
    ):
        full_range = OptionalSet(set=set(range(num_frames))
            if num_frames is not None else None)

        self.update(frame_range.intersection(full_range))

    def intersection(self, other: OptionalSet):
        return FrameRange(self.frame_range.intersection(other))

    def update(self, frame_range: OptionalSet):
        assert frame_range.set is not None

        self.frame_range = frame_range

        # Continuous index of all frames in the range.
        all_frames = sorted(self.frame_range.set)
        self.index_to_frame = {i: f for i, f in enumerate(all_frames)}

    def frames(self):
        return sorted(self.index_to_frame.values())

    def __len__(self):
        return len(self.index_to_frame)


def parse_frame_range(frame_range_str: str) -> NamedOptionalSet:
    """
    Create a frame range from a string, e.g.: 1-10,15,21-40,51-62.
    """
    if len(frame_range_str) == 0:
        return NamedOptionalSet(name=frame_range_str, set=OptionalSet())

    range_strs = frame_range_str.split(',')

    def parse_sub_range(sub_range_str: str):
        splits = [int(s) for s in sub_range_str.split('-', maxsplit=1)]
        if len(splits) == 1:
            return splits

        start, end = splits
        assert start <= end
        return range(start, end + 1)

    frame_range = set()
    for range_str in range_strs:
        frame_range.update(parse_sub_range(range_str))

    # Convert the range to a friendly string representation, e.g.,
    # 6,6,5,8,0,2-4,5-6,10,9 -> "0,2-6,8-10"
    it = iter(sorted(frame_range))

    ranges = []
    start = next(it)
    last_index = start

    def add_range(ranges):
        if last_index == start:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{last_index}")

    for i in it:
        if i < 0:
            raise ValueError("Frame indices must be positive.")
        assert(i > last_index)
        if i - last_index > 1:
            add_range(ranges)
            start = i
        last_index = i
    add_range(ranges)

    name = ",".join(ranges)

    return NamedOptionalSet(name=name, set=OptionalSet(frame_range))

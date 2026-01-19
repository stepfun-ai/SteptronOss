import json
import os
from functools import cached_property

import boto3
import megfile
import numpy as np
from megfile.s3_path import get_endpoint_url


def safediv(n, d):
    q, r = divmod(n, d)
    assert r == 0
    return q


def normalize_slice(size, start, stop, step):
    """
    ```
    start, stop, step = normalize_slice(len(x), start, stop, step)
    i = start
    while i != stop:
        assert 0 <= i < size
        x[i]
        i += step
    ```
    """
    assert step != 0
    if step is None:
        step = 1
    if stop is None:
        if step > 0:
            stop = size
        else:
            stop = -1
    elif stop < 0:
        stop += size
    if start is None:
        if step > 0:
            start = 0
        else:
            start = size - 1
    elif start < 0:
        start += size
    if step > 0:
        start = min(max(start, 0), size)
        stop = min(max(stop, 0), size)
        num_elems = (stop - start - 1) // step + 1
    else:
        start = min(max(start, -1), size - 1)
        stop = min(max(stop, -1), size - 1)
        num_elems = (stop - start + 1) // step + 1
    num_elems = max(num_elems, 0)
    stop = start + num_elems * step
    return start, stop, step


def compute_span(size, start, stop, step):
    """
    ```
    span_start, span_stop = compute_span(len(x), start, stop, step)
    assert 0 <= span_start < len(x) and 0 <= span_stop <= len(x) or span_start == span_stop
    assert x[slice(span_start, span_stop)][::step] == x[slice(start, stop, step)]
    ```
    """
    start, stop, step = normalize_slice(size, start, stop, step)
    num_elems = safediv(stop - start, step)
    if not num_elems:
        return start, start
    first = start
    last = start + step * (num_elems - 1)
    if step > 0:
        return first, last + 1
    else:
        return last, first + 1


def _indexed_bin_writer(path, prefix="", cfg=None):
    def open_output(file, mode="wb"):
        return megfile.smart_open(megfile.smart_path_join(path, file), mode)

    offset = 0
    offsets = []
    sizes = []

    size = None
    with open_output(prefix + "data.bin") as data_file:
        while True:
            data = yield size
            if data is None:
                break
            data = memoryview(data)
            size = data_file.write(data)
            assert size == data.nbytes
            offsets.append(offset)
            sizes.append(size)
            offset += size

    with open_output(prefix + "index.npz") as index_file:
        np.savez_compressed(
            index_file,
            offsets=np.array(offsets, dtype="<u8"),
            sizes=np.array(sizes, dtype="<u8"),
        )
    if cfg:
        with open_output("cfg.json", "w") as cfg_file:
            json.dump(cfg, cfg_file)
    with open_output(".success") as success_file:
        pass


class IndexedBinWriter:
    def __init__(self, path, prefix="", cfg=None) -> None:
        self._writer = _indexed_bin_writer(path, prefix, cfg=cfg)
        self._writer.send(None)
        self._size = 0

    def write(self, data):
        assert data is not None
        size = self._writer.send(data)
        self._size += size
        return size

    def tell(self):
        return self._size

    def close(self):
        if self._writer is not None:
            try:
                self._writer.send(None)
            except StopIteration:
                self._writer = None
            else:
                assert 0

    def abort(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer is not None:
            if exc_type is None:
                self.close()
            else:
                self.abort()


def readexactly(file, size):
    data = memoryview(bytearray(size))
    p = 0
    while p < size:
        n = file.readinto(data[p:])
        assert n > 0
        p += n
    return bytes(data)


def pread(file, offset, size):
    data = bytearray(size)
    bytes_read = 0
    while bytes_read < size:
        chunk = os.pread(file.fileno(), size - bytes_read, offset + bytes_read)
        if not chunk:
            raise IOError(f"{file.name}: Unexpected end of file")
        data[bytes_read : bytes_read + len(chunk)] = chunk
        bytes_read += len(chunk)
    return data


class FileReader:
    file = None

    def __init__(self, path) -> None:
        self.path = path

    def read(self, offset, size):
        # lazy open to be fork-friendly
        if self.file is None:
            self.file = open(self.path, "rb", buffering=0)
        # self.file.seek(offset)
        # return readexactly(self.file, size)
        return pread(self.file, offset, size)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("file", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class S3Reader:
    client = None

    def __init__(self, path: str) -> None:
        assert path.startswith("s3://")
        path = path[len("s3://") :]
        self.bucket, self.key = path.split("/", 1)

    def read(self, offset, size):
        offset = int(offset)
        size = int(size)
        assert offset >= 0
        assert size >= 0
        if size == 0:
            return b""

        if self.client is None:
            self.client = boto3.client("s3", endpoint_url=get_endpoint_url())

        response = self.client.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={offset}-{offset+size-1}")
        assert int(response["ContentLength"]) == size
        return response["Body"].read()

    def close(self):
        # S3 client doesn't need explicit closing, but we can clear the reference
        self.client = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)

    def __setstate__(self, state):
        self.__dict__ = state


def AutoReader(path: str):
    if path.startswith("s3://"):
        return S3Reader(path)
    else:
        return FileReader(path)


class IndexedBinReader:
    def __init__(self, path, dtype=None, prefix="") -> None:
        self.path = path
        self.dtype = dtype
        if self.dtype is not None:
            self.dtype = np.dtype(self.dtype)
        self.prefix = prefix

        self._load_cfg()
        assert megfile.smart_exists(self._get_file(".success")), path
        self.data_reader = AutoReader(self._get_file("data.bin"))

        self._offsets = None
        self._sizes = None
        self._load_index()
        # self._check()

    @cached_property
    def _itemsize(self):
        if self.dtype is None:
            return 1
        else:
            return self.dtype.itemsize

    def _check(self):
        expected_size = sum(self.sizes) * self._itemsize
        actual_size = megfile.smart_getsize(self._get_file("data.bin"))
        if expected_size != actual_size:
            print(f"Bin: {self.path} broken!\nexpect {expected_size} get {actual_size}")

    def _load_cfg(self):
        potential_cfg = self._get_file("cfg.json")
        if megfile.smart_exists(potential_cfg):
            with megfile.smart_open(potential_cfg, "r") as f:
                cfg = json.load(f)
            if "dtype" in cfg:
                self.dtype = np.dtype(cfg["dtype"])

    @property
    def offsets(self):
        if self._offsets is None:
            self._load_index()
        return self._offsets

    @property
    def sizes(self):
        if self._sizes is None:
            self._load_index()
        return self._sizes

    def _load_index(self):
        index = np.load(megfile.smart_open(self._get_file("index.npz"), "rb"))
        self._offsets, r = np.divmod(index["offsets"], self._itemsize)
        assert not np.any(r)
        self._sizes, r = np.divmod(index["sizes"], self._itemsize)
        assert not np.any(r)

    def _get_file(self, name):
        if name == ".success":
            return megfile.smart_path_join(self.path, name)
        else:
            return megfile.smart_path_join(self.path, self.prefix + name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("offsets", None)
        state.pop("sizes", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._load_index()

    def _read_span(self, i, start, stop):
        size = stop - start
        data = self.data_reader.read((int(self.offsets[i]) + start) * self._itemsize, size * self._itemsize)
        if self.dtype:
            return np.frombuffer(data, dtype=self.dtype)
        else:
            return data

    def _get_elem(self, i, j):
        size = int(self.sizes[i])
        if j < 0:
            j += size
        assert 0 <= j < size
        return self._read_span(i, j, j + 1)[0]

    def _get_slice(self, i, start, stop, step):
        span = compute_span(int(self.sizes[i]), start, stop, step)
        data = self._read_span(i, *span)
        return data[::step]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, arg):
        start, stop, step = None, None, None
        if isinstance(arg, tuple):
            if len(arg) == 1:
                (i,) = arg
            else:
                i, j = arg
                if j is Ellipsis:
                    pass
                elif isinstance(j, slice):
                    start, stop, step = j.start, j.stop, j.step
                else:
                    return self._get_elem(i, j)
        else:
            i = arg
        return self._get_slice(i, start, stop, step)

    def close(self):
        if hasattr(self, "data_reader") and self.data_reader is not None:
            if hasattr(self.data_reader, "close"):
                self.data_reader.close()
            self.data_reader = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    with IndexedBinWriter("./test_bin_file/") as writer:
        for i in range(10):
            writer.write((f"{i}" * i).encode())

    reader = IndexedBinReader("./test_bin_file/")
    print(len(reader))
    for i in range(len(reader)):
        print(reader[i])

    with IndexedBinWriter("./test_bin_file/") as writer:
        for i in range(10):
            writer.write(np.array([1, 2, 3], dtype=np.uint16))

    reader = IndexedBinReader("./test_bin_file/", dtype=np.uint16)
    print(len(reader))
    for i in range(len(reader)):
        print(reader[i, slice(0, 3, 2)])

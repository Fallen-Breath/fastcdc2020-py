import math
import mmap
import os
from pathlib import Path
from typing import Callable, Union, Optional

from pyfastcdc.common import BinaryStreamReader

ReadintoFunc = Callable[[memoryview], int]


def logarithm2(value: int) -> int:
	# The same as the nlfiedler/fastcdc-rs v4.0.1 Rust implementation: (value as f64).log2().round() as u32
	if value <= 0:
		raise ValueError('value must be positive')
	return math.floor(math.log2(value) + 0.5)


def create_memoryview_from_buffer(buf: Union[bytes, bytearray, memoryview]) -> memoryview:
	if isinstance(buf, (bytes, bytearray, memoryview)):
		return memoryview(buf)
	raise TypeError('buf must be bytes or bytearray')


def create_readinto_func(stream: BinaryStreamReader) -> ReadintoFunc:
	readinto_func: ReadintoFunc = getattr(stream, 'readinto', None)
	if readinto_func is not None and callable(readinto_func):
		return readinto_func

	read_func = getattr(stream, 'read', None)
	if read_func is not None and callable(read_func):
		def readinto_using_read(dest_buf: memoryview) -> int:
			read_buf = read_func(len(dest_buf))
			dest_buf[:len(read_buf)] = read_buf
			return len(read_buf)

		return readinto_using_read

	raise TypeError('stream must be readable')


class MmapFile:
	def __init__(self, file_path: Union[str, bytes, Path]):
		self.__mmap_obj: Optional[mmap.mmap] = None
		self.__data = memoryview(b'')
		self.__open(file_path)

	def __open(self, file_path: Union[str, bytes, Path]):
		file_size = os.path.getsize(file_path)
		if file_size == 0:
			return

		with open(file_path, 'rb') as f:
			self.__mmap_obj = mmap.mmap(f.fileno(), length=file_size, access=mmap.ACCESS_READ)
			self.__data = memoryview(self.__mmap_obj)

	@property
	def data(self) -> memoryview:
		return self.__data


def create_mmap_from_file(file_path: Union[str, bytes, Path]) -> MmapFile:
	return MmapFile(file_path)

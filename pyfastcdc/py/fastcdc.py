import array
from pathlib import Path
from typing import Optional, ClassVar, Union, Iterator

from pyfastcdc import utils
from pyfastcdc.common import BinaryStreamReader, NormalizedChunking
from pyfastcdc.py.chunk import Chunk
from pyfastcdc.py.constants import MASKS, GEAR, GEAR_LS
from pyfastcdc.utils import ReadintoFunc

_UINT64_MASK = (1 << 64) - 1


class _Config:
	avg_size: int
	min_size: int
	max_size: int
	normalized_chunking: NormalizedChunking
	mask_s: int
	mask_l: int
	mask_s_ls: int
	mask_l_ls: int
	gear: 'array.array[int]'
	gear_ls: 'array.array[int]'

	def __init__(
			self,
			avg_size: int,
			min_size: int,
			max_size: int,
			normalized_chunking: NormalizedChunking,
			mask_s: int,
			mask_l: int,
			mask_s_ls: int,
			mask_l_ls: int,
			gear: 'array.array[int]',
			gear_ls: 'array.array[int]',
	):
		self.avg_size = avg_size
		self.min_size = min_size
		self.max_size = max_size
		self.normalized_chunking = normalized_chunking
		self.mask_s = mask_s
		self.mask_l = mask_l
		self.mask_s_ls = mask_s_ls
		self.mask_l_ls = mask_l_ls
		self.gear = gear
		self.gear_ls = gear_ls


# docstrings are in pyfastcdc/__init__.pyi
class FastCDC:
	MIN_SIZE_LOWER_BOUND: ClassVar[int] = 64
	AVG_SIZE_LOWER_BOUND: ClassVar[int] = 256
	MAX_SIZE_LOWER_BOUND: ClassVar[int] = 1024
	MIN_SIZE_UPPER_BOUND: ClassVar[int] = 1 * 1048576
	AVG_SIZE_UPPER_BOUND: ClassVar[int] = 4 * 1048576
	MAX_SIZE_UPPER_BOUND: ClassVar[int] = 16 * 1048576

	def __init__(
			self,
			avg_size: int = 16384,
			*,
			min_size: Optional[int] = None,
			max_size: Optional[int] = None,
			normalized_chunking: NormalizedChunking = 1,
			seed: int = 0,
	):
		if min_size is None:
			min_size = avg_size // 4
		if max_size is None:
			max_size = avg_size * 4
		if not (self.AVG_SIZE_LOWER_BOUND <= avg_size <= self.AVG_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {avg_size} is out of range [{self.AVG_SIZE_LOWER_BOUND}, {self.AVG_SIZE_UPPER_BOUND}]')
		if not (self.MIN_SIZE_LOWER_BOUND <= min_size <= self.MIN_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {min_size} is out of range [{self.MIN_SIZE_LOWER_BOUND}, {self.MIN_SIZE_UPPER_BOUND}]')
		if not (self.MAX_SIZE_LOWER_BOUND <= max_size <= self.MAX_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {max_size} is out of range [{self.MAX_SIZE_LOWER_BOUND}, {self.MAX_SIZE_UPPER_BOUND}]')
		if not (min_size <= avg_size <= max_size):
			raise ValueError(f'avg_size {avg_size} is out of range [{min_size}, {max_size}]')
		if not (0 <= normalized_chunking <= 3):
			raise ValueError(f'normalized_chunking {normalized_chunking} is out of range [0, 3]')

		bits = utils.logarithm2(avg_size)
		mask_s = MASKS[bits + normalized_chunking]
		mask_l = MASKS[bits - normalized_chunking]
		mask_s_ls = (mask_s << 1) & _UINT64_MASK
		mask_l_ls = (mask_l << 1) & _UINT64_MASK

		gear = GEAR
		gear_ls = GEAR_LS
		if seed > 0:
			seed &= _UINT64_MASK
			seed_ls = (seed << 1) & _UINT64_MASK
			gear = array.array('Q', [(x ^ seed) & _UINT64_MASK for x in GEAR])
			gear_ls = array.array('Q', [(x ^ seed_ls) & _UINT64_MASK for x in GEAR_LS])

		self.config = _Config(
			avg_size=avg_size,
			min_size=min_size,
			max_size=max_size,
			normalized_chunking=normalized_chunking,
			mask_s=mask_s,
			mask_l=mask_l,
			mask_s_ls=mask_s_ls,
			mask_l_ls=mask_l_ls,
			gear=gear,
			gear_ls=gear_ls,
		)

	def cut_buf(self, buf: Union[bytes, bytearray, memoryview]) -> Iterator[Chunk]:
		return BufferChunker(self.config, utils.create_memoryview_from_buffer(buf))

	def cut_file(self, file_path: Union[str, bytes, Path]) -> Iterator[Chunk]:
		return FileMmapChunker(self.config, file_path)

	def cut_stream(self, stream: BinaryStreamReader) -> Iterator[Chunk]:
		return StreamChunker(self.config, utils.create_readinto_func(stream))

	@property
	def avg_size(self) -> int:
		return self.config.avg_size

	@property
	def min_size(self) -> int:
		return self.config.min_size

	@property
	def max_size(self) -> int:
		return self.config.max_size


class _CutResult:
	gear_hash: int
	cut_offset: int

	def __init__(self, gear_hash: int, cut_offset: int):
		self.gear_hash = gear_hash
		self.cut_offset = cut_offset


def _cut_gear(config: _Config, buf: memoryview) -> _CutResult:
	remaining = len(buf)
	if remaining <= config.min_size:
		return _CutResult(0, remaining)
	center = config.avg_size
	if remaining > config.max_size:
		remaining = config.max_size
	elif remaining < center:
		center = remaining

	# speed up variable lookup
	gear = config.gear
	gear_ls = config.gear_ls
	mask_s = config.mask_s
	mask_l = config.mask_l
	mask_s_ls = config.mask_s_ls
	mask_l_ls = config.mask_l_ls
	mask64 = _UINT64_MASK

	gear_hash = 0
	start_pos = (config.min_size // 2) * 2
	mid_pos = (center // 2) * 2
	end_pos = (remaining // 2) * 2

	for pos in range(start_pos, mid_pos, 2):
		gear_hash = ((gear_hash << 2) + gear_ls[buf[pos]]) & mask64
		if not (gear_hash & mask_s_ls):
			return _CutResult(gear_hash, pos)
		gear_hash = (gear_hash + gear[buf[pos + 1]]) & mask64
		if not (gear_hash & mask_s):
			return _CutResult(gear_hash, pos + 1)

	for pos in range(mid_pos, end_pos, 2):
		gear_hash = ((gear_hash << 2) + gear_ls[buf[pos]]) & mask64
		if not (gear_hash & mask_l_ls):
			return _CutResult(gear_hash, pos)
		gear_hash = (gear_hash + gear[buf[pos + 1]]) & mask64
		if not (gear_hash & mask_l):
			return _CutResult(gear_hash, pos + 1)

	return _CutResult(gear_hash, remaining)


class BufferChunker(Iterator[Chunk]):
	def __init__(self, config: _Config, buf: memoryview):
		self.config = config
		self.buf = buf
		self.offset = 0

	def __next__(self) -> Chunk:
		if self.offset >= len(self.buf):
			raise StopIteration()

		res = _cut_gear(self.config, self.buf[self.offset:])
		end_pos = self.offset + res.cut_offset

		chunk = Chunk(
			offset=self.offset,
			length=res.cut_offset,
			data=self.buf[self.offset:end_pos],
			gear_hash=res.gear_hash,
		)
		self.offset += res.cut_offset
		return chunk


class FileMmapChunker(BufferChunker):
	def __init__(self, config: _Config, file_path: Union[str, bytes, Path]):
		self.mmap_file = utils.create_mmap_from_file(file_path)
		super().__init__(config, self.mmap_file.data)


class StreamChunker(Iterator[Chunk]):
	def __init__(self, config: _Config, readinto_func: ReadintoFunc):
		self.config = config
		self.readinto_func = readinto_func

		self.offset = 0
		self.last_chunk_len = 0
		self.eof = False

		self.buf_capacity = self.config.max_size + max(8 * 1024, self.config.max_size)
		self.buf = bytearray(self.buf_capacity)
		self.buf_read_len = 0
		self.buf_write_len = 0

	def __next__(self) -> Chunk:
		if self.last_chunk_len > 0:
			self.__release_last_chunk()

		remaining_buf_len = self.buf_write_len - self.buf_read_len
		if not self.eof and remaining_buf_len < self.config.max_size:
			while self.buf_write_len < self.buf_capacity:
				n_read = self.readinto_func(memoryview(self.buf)[self.buf_write_len:])
				if n_read == 0:
					break
				self.buf_write_len += n_read

		remaining_buf_len = self.buf_write_len - self.buf_read_len
		if remaining_buf_len == 0:
			raise StopIteration()

		res = _cut_gear(self.config, memoryview(self.buf)[self.buf_read_len:self.buf_write_len])
		chunk_len = res.cut_offset
		if res.cut_offset == 0:  # last part of the file
			chunk_len = remaining_buf_len

		self.last_chunk_len = chunk_len
		return Chunk(
			offset=self.offset,
			length=chunk_len,
			data=memoryview(self.buf)[self.buf_read_len:self.buf_read_len + chunk_len],
			gear_hash=res.gear_hash,
		)

	def __release_last_chunk(self):
		self.buf_read_len += self.last_chunk_len
		self.offset += self.last_chunk_len
		self.last_chunk_len = 0
		if self.buf_read_len > self.buf_write_len:
			raise AssertionError(f'buf_read_len {self.buf_read_len} is greater than buf_write_len {self.buf_write_len}')

		remaining_buf_len = self.buf_write_len - self.buf_read_len
		if remaining_buf_len < self.config.max_size:
			self.buf[:remaining_buf_len] = self.buf[self.buf_read_len:self.buf_write_len]
			self.buf_read_len = 0
			self.buf_write_len = remaining_buf_len

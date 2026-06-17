from pathlib import Path
from typing import Optional, Union, Iterator

import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.string cimport memmove

from pyfastcdc import utils
from pyfastcdc.common import BinaryStreamReader, NormalizedChunking
from pyfastcdc.cy.chunk cimport Chunk
from pyfastcdc.cy.constants cimport GEAR, GEAR_LS, MASKS
from pyfastcdc.utils import ReadintoFunc

cdef struct _Config:
	uint32_t avg_size
	uint32_t min_size
	uint32_t max_size
	uint64_t mask_s
	uint64_t mask_l
	uint64_t mask_s_ls
	uint64_t mask_l_ls
	const uint64_t* gear
	const uint64_t* gear_ls


cdef uint64_t MIN_SIZE_LOWER_BOUND = 64
cdef uint64_t AVG_SIZE_LOWER_BOUND = 256
cdef uint64_t MAX_SIZE_LOWER_BOUND = 1024
cdef uint64_t MIN_SIZE_UPPER_BOUND = 1 * 1048576
cdef uint64_t AVG_SIZE_UPPER_BOUND = 4 * 1048576
cdef uint64_t MAX_SIZE_UPPER_BOUND = 16 * 1048576


# docstrings are in pyfastcdc/__init__.pyi
cdef class FastCDC:
	cdef _Config config
	cdef uint64_t* gear_holder
	cdef uint64_t* gear_holder_ls

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
		if not (AVG_SIZE_LOWER_BOUND <= avg_size <= AVG_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {avg_size} is out of range [{AVG_SIZE_LOWER_BOUND}, {AVG_SIZE_UPPER_BOUND}]')
		if not (MIN_SIZE_LOWER_BOUND <= min_size <= MIN_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {min_size} is out of range [{MIN_SIZE_LOWER_BOUND}, {MIN_SIZE_UPPER_BOUND}]')
		if not (MAX_SIZE_LOWER_BOUND <= max_size <= MAX_SIZE_UPPER_BOUND):
			raise ValueError(f'min_size {max_size} is out of range [{MAX_SIZE_LOWER_BOUND}, {MAX_SIZE_UPPER_BOUND}]')
		if not (min_size <= avg_size <= max_size):
			raise ValueError(f'avg_size {avg_size} is out of range [{min_size}, {max_size}]')
		if not (0 <= normalized_chunking <= 3):
			raise ValueError(f'normalized_chunking {normalized_chunking} is out of range [0, 3]')

		self.config.avg_size = avg_size
		self.config.min_size = min_size
		self.config.max_size = max_size

		bits = utils.logarithm2(avg_size)
		self.config.mask_s = MASKS[bits + normalized_chunking]
		self.config.mask_l = MASKS[bits - normalized_chunking]
		self.config.mask_s_ls = self.config.mask_s << 1
		self.config.mask_l_ls = self.config.mask_l << 1

		self.config.gear = GEAR
		self.config.gear_ls = GEAR_LS
		self.gear_holder = NULL
		self.gear_holder_ls = NULL

		cdef uint64_t seed_ls = seed << 1
		if seed > 0:
			self.gear_holder = <uint64_t*>PyMem_Malloc(256 * sizeof(uint64_t))
			self.gear_holder_ls = <uint64_t*>PyMem_Malloc(256 * sizeof(uint64_t))
			for i in range(256):
				self.gear_holder[i] = GEAR[i] ^ seed
				self.gear_holder_ls[i] = GEAR_LS[i] ^ seed_ls
			self.config.gear = self.gear_holder
			self.config.gear_ls = self.gear_holder_ls

	def __dealloc__(self):
		if self.gear_holder:
			PyMem_Free(self.gear_holder)
			self.gear_holder = NULL
		if self.gear_holder_ls:
			PyMem_Free(self.gear_holder_ls)
			self.gear_holder_ls = NULL

	def cut_buf(self, buf: Union[bytes, bytearray, memoryview]) -> Iterator[Chunk]:
		return BufferChunker(self, utils.create_memoryview_from_buffer(buf))

	def cut_file(self, file_path: Union[str, bytes, Path]) -> Iterator[Chunk]:
		return FileMmapChunker(self, file_path)

	def cut_stream(self, stream: BinaryStreamReader) -> Iterator[Chunk]:
		return StreamChunker(self, utils.create_readinto_func(stream))

	@property
	def avg_size(self) -> int:
		return self.config.avg_size

	@property
	def min_size(self) -> int:
		return self.config.min_size

	@property
	def max_size(self) -> int:
		return self.config.max_size


cdef struct _CutResult:
	uint64_t gear_hash
	uint64_t cut_offset


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _CutResult _cut_gear(const _Config* config, const uint8_t* buf, uint64_t buf_len) noexcept nogil:
	cdef uint64_t remaining = buf_len
	if remaining <= config.min_size:
		return _CutResult(0, remaining)
	cdef uint64_t center = config.avg_size
	if remaining > config.max_size:
		remaining = config.max_size
	elif remaining < center:
		center = remaining

	cdef uint64_t pos = 0
	cdef uint64_t gear_hash = 0
	cdef uint64_t start_pos = (config.min_size // 2) * 2
	cdef uint64_t mid_pos = (center // 2) * 2
	cdef uint64_t end_pos = (remaining // 2) * 2

	cdef uint64_t mask_s = config.mask_s
	cdef uint64_t mask_s_ls = config.mask_s_ls
	cdef uint64_t mask_l = config.mask_l
	cdef uint64_t mask_l_ls = config.mask_l_ls
	cdef const uint64_t* gear_ptr = config.gear
	cdef const uint64_t* gear_ls_ptr = config.gear_ls

	for pos in range(start_pos, mid_pos, 2):
		gear_hash = (gear_hash << 2) + gear_ls_ptr[buf[pos]]
		if (gear_hash & mask_s_ls) == 0:
			return _CutResult(gear_hash, pos)
		gear_hash = gear_hash + gear_ptr[buf[pos + 1]]
		if (gear_hash & mask_s) == 0:
			return _CutResult(gear_hash, pos + 1)

	for pos in range(mid_pos, end_pos, 2):
		gear_hash = ((gear_hash << 2) + gear_ls_ptr[buf[pos]])
		if (gear_hash & mask_l_ls) == 0:
			return _CutResult(gear_hash, pos)
		gear_hash = gear_hash + gear_ptr[buf[pos + 1]]
		if (gear_hash & mask_l) == 0:
			return _CutResult(gear_hash, pos + 1)

	return _CutResult(gear_hash, remaining)


cdef class BufferChunker:
	cdef object fastcdc
	cdef const _Config * config
	cdef memoryview buf
	cdef const uint8_t[:] buf_view
	cdef uint64_t buf_capacity
	cdef uint64_t offset

	def __init__(self, fastcdc: FastCDC, buf: memoryview):
		self.fastcdc = fastcdc  # keep ref
		self.config = &fastcdc.config
		self.buf = buf
		self.buf_view = buf
		self.buf_capacity = len(buf)
		self.offset = 0

	def __next__(self) -> Chunk:
		if self.offset >= self.buf_capacity:
			raise StopIteration()

		cdef const uint8_t* remaining_buf = &self.buf_view[0] + self.offset
		cdef uint64_t remaining_len = self.buf_capacity - self.offset
		cdef _CutResult res
		with nogil:
			res = _cut_gear(self.config, remaining_buf, remaining_len)
		cdef uint64_t end_pos = self.offset + res.cut_offset

		chunk = Chunk._cy_create(
			offset=self.offset,
			length=res.cut_offset,
			data=self.buf[self.offset:end_pos],
			gear_hash=res.gear_hash,
		)
		self.offset += res.cut_offset
		return chunk

	def __iter__(self):
		return self


cdef class FileMmapChunker(BufferChunker):
	cdef object mmap_file

	def __init__(self, fastcdc: FastCDC, file_path: Union[str, bytes, Path]):
		self.mmap_file = utils.create_mmap_from_file(file_path)
		BufferChunker.__init__(self, fastcdc, self.mmap_file.data)


cdef class StreamChunker:
	cdef object fastcdc
	cdef const _Config * config
	cdef object readinto_func
	cdef uint64_t max_size

	cdef uint64_t offset
	cdef uint64_t last_chunk_len
	cdef uint8_t eof

	cdef uint64_t buf_capacity
	cdef bytearray buf_obj
	cdef memoryview buf_obj_mv
	cdef uint8_t[:] buf_view
	cdef uint64_t buf_read_len
	cdef uint64_t buf_write_len

	def __init__(self, fastcdc: FastCDC, readinto_func: ReadintoFunc):
		self.fastcdc = fastcdc  # keep ref
		self.config = &fastcdc.config
		self.readinto_func = readinto_func
		self.max_size = fastcdc.config.max_size

		self.offset = 0
		self.last_chunk_len = 0
		self.eof = 0

		self.buf_capacity = self.max_size + max(8 * 1024u, self.config.max_size)
		self.buf_obj = bytearray(self.buf_capacity)
		self.buf_obj_mv = memoryview(self.buf_obj)
		self.buf_view = self.buf_obj
		self.buf_read_len = 0
		self.buf_write_len = 0

	def __next__(self) -> Chunk:
		cdef uint64_t remaining_buf_len = 0
		cdef Py_ssize_t n_read = 0
		cdef uint8_t* buf_ptr = &self.buf_view[0]

		if self.last_chunk_len > 0:
			self.__release_last_chunk()

		remaining_buf_len = self.buf_write_len - self.buf_read_len
		if not self.eof and remaining_buf_len < self.max_size:
			while self.buf_write_len < self.buf_capacity:
				n_read = self.readinto_func(self.buf_obj_mv[self.buf_write_len:])
				if n_read <= 0:
					self.eof = 1
					break
				self.buf_write_len += n_read

		remaining_buf_len = self.buf_write_len - self.buf_read_len
		if remaining_buf_len == 0:
			raise StopIteration()

		cdef _CutResult res
		with nogil:
			res = _cut_gear(self.config, buf_ptr + self.buf_read_len, remaining_buf_len)

		cdef uint64_t chunk_len = res.cut_offset
		if chunk_len == 0:  # last part of the file
			chunk_len = remaining_buf_len

		self.last_chunk_len = chunk_len
		return Chunk(
			offset=self.offset,
			length=chunk_len,
			data=self.buf_obj_mv[self.buf_read_len:self.buf_read_len + chunk_len],
			gear_hash=res.gear_hash,
		)

	cdef __release_last_chunk(self):
		cdef uint8_t* buf_ptr = &self.buf_view[0]

		self.buf_read_len += self.last_chunk_len
		self.offset += self.last_chunk_len
		self.last_chunk_len = 0
		if self.buf_read_len > self.buf_write_len:
			raise AssertionError(f'buf_read_len {self.buf_read_len} is greater than buf_write_len {self.buf_write_len}')

		cdef uint64_t remaining_buf_len = self.buf_write_len - self.buf_read_len
		if remaining_buf_len < self.max_size:
			memmove(buf_ptr, buf_ptr + self.buf_read_len, remaining_buf_len)
			self.buf_read_len = 0
			self.buf_write_len = remaining_buf_len

	def __iter__(self):
		return self

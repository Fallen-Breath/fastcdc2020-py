import pytest

from pyfastcdc import NormalizedChunking
from tests.utils import FastCDCType


# NOTE: The .data field of Chunks from cut_buf() are always valid

class TestParameterValidation:
	def test_valid_avg_size(self, fastcdc_impl: FastCDCType):
		assert fastcdc_impl(avg_size=256).avg_size == 256
		assert fastcdc_impl(avg_size=4 * 1024 * 1024).avg_size == 4 * 1024 * 1024
		assert fastcdc_impl(avg_size=16384).avg_size == 16384

	def test_invalid_avg_size(self, fastcdc_impl: FastCDCType):
		with pytest.raises(ValueError, match="out of range"):
			fastcdc_impl(avg_size=63)
		with pytest.raises(ValueError, match="out of range"):
			fastcdc_impl(avg_size=5 * 1024 * 1024)

	def test_custom_min_max_sizes(self, fastcdc_impl: FastCDCType):
		cdc = fastcdc_impl(avg_size=16384, min_size=4096, max_size=65536)
		assert cdc.min_size == 4096
		assert cdc.max_size == 65536

		cdc = fastcdc_impl(avg_size=16384)
		assert cdc.min_size == 16384 // 4
		assert cdc.max_size == 16384 * 4

	def test_normalized_chunking_values(self, fastcdc_impl: FastCDCType):
		value: NormalizedChunking
		for value in [0, 1, 2, 3]:
			cdc = fastcdc_impl(avg_size=8192, normalized_chunking=value)
			assert cdc is not None
		with pytest.raises(ValueError):
			fastcdc_impl(avg_size=8192, normalized_chunking=4)  # type: ignore


class TestEdgeCases:
	def test_empty_data(self, fastcdc_instance):
		chunks = list(fastcdc_instance.cut_buf(b''))
		assert len(chunks) == 0

	def test_smaller_than_min_size(self, fastcdc_impl: FastCDCType):
		data = b'x' * 100
		cdc = fastcdc_impl(avg_size=16384, min_size=4096)
		chunks = list(cdc.cut_buf(data))

		assert len(chunks) == 1
		assert chunks[0].length == 100
		assert chunks[0].offset == 0
		assert bytes(chunks[0].data) == data

	def test_exactly_min_size(self, fastcdc_impl: FastCDCType):
		cdc = fastcdc_impl(avg_size=16384, min_size=4096)
		data = b'x' * 4096
		chunks = list(cdc.cut_buf(data))

		assert len(chunks) == 1
		assert chunks[0].length == 4096

	def test_larger_than_max_size(self, fastcdc_impl: FastCDCType):
		data = b'x' * (200 * 1024)
		cdc = fastcdc_impl(avg_size=16384, max_size=65536)
		chunks = list(cdc.cut_buf(data))

		assert len(chunks) > 1
		assert all(chunk.length <= 65536 for chunk in chunks)
		total_size = sum(chunk.length for chunk in chunks)
		assert total_size == len(data)

	def test_single_byte(self, fastcdc_instance):
		chunks = list(fastcdc_instance.cut_buf(b'x'))
		assert len(chunks) == 1
		assert chunks[0].length == 1
		assert bytes(chunks[0].data) == b'x'

	def test_repeated_pattern(self, fastcdc_instance):
		data = b'abcd' * 1000
		chunks = list(fastcdc_instance.cut_buf(data))

		total_size = sum(chunk.length for chunk in chunks)
		assert total_size == len(data)

		reconstructed = b''.join(bytes(chunk.data) for chunk in chunks)
		assert reconstructed == data

	def test_nonexistent_file(self, fastcdc_instance):
		with pytest.raises(FileNotFoundError):
			list(fastcdc_instance.cut_file('/nonexistent/file/path'))


class TestAvgSize:
	def test_cut_with_different_avg_sizes(self, fastcdc_impl: FastCDCType, random_data_1m: bytes):
		avg_sizes = [1024, 4096, 16384, 65536]
		chunk_counts = []

		for avg_size in avg_sizes:
			cdc = fastcdc_impl(avg_size=avg_size)
			chunks = list(cdc.cut_buf(random_data_1m))
			chunk_counts.append(len(chunks))

		# larger avg_size means fewer chunks
		for i in range(1, len(chunk_counts)):
			assert chunk_counts[i] <= chunk_counts[i - 1]

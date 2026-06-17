import pytest

from pyfastcdc import utils


class TestLogarithm2:
	def test_negative_input(self):
		with pytest.raises(ValueError):
			utils.logarithm2(-1)

	def test_zero(self):
		assert utils.logarithm2(0) == 0

	@pytest.mark.parametrize('result, lower, upper', [
		(0, 0, 2),
		(1, 2, 3),
		(2, 3, 6),
		(3, 6, 12),
		(4, 12, 23),
		(5, 23, 46),
		(6, 46, 91),
		(7, 91, 182),
		(8, 182, 363),
		(9, 363, 725),
		(10, 725, 1449),
		(11, 1449, 2897),
		(12, 2897, 5793),
		(13, 5793, 11586),
		(14, 11586, 23171),
		(15, 23171, 46341),
		(16, 46341, 92682),
		(17, 92682, 185364),
		(18, 185364, 370728),
		(19, 370728, 741456),
		(20, 741456, 1482911),
		(21, 1482911, 2965821),
		(22, 2965821, 5931642),
		(23, 5931642, 11863284),
		(24, 11863284, 23726567),
		(25, 23726567, 47453133),
		(26, 47453133, 94906266),
		(27, 94906266, 189812532),
		(28, 189812532, 379625063),
		(29, 379625063, 759250125),
		(30, 759250125, 1518500250),
		(31, 1518500250, 3037000500),
		(32, 3037000500, 6074001000),
	])
	def test_logarithm2_boundaries(self, result: int, lower: int, upper: int):
		if upper - lower < 30000:
			for i in range(lower, upper):
				assert utils.logarithm2(i) == result
		else:
			for i in range(lower, lower + 10000):
				assert utils.logarithm2(i) == result
			for i in range(upper - 10000, upper):
				assert utils.logarithm2(i) == result
			for i in range(100):
				mid = lower + (upper - lower) * i // 100
				assert utils.logarithm2(mid) == result
		assert utils.logarithm2(upper) == result + 1

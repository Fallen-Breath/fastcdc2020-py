__version__ = '0.3.0'

__all__ = [
	'BinaryStreamReader',
	'Chunk',
	'FastCDC',
	'NormalizedChunking',
]

from pyfastcdc.common import (
	BinaryStreamReader,
	NormalizedChunking,
)

try:
	from pyfastcdc.cy import FastCDC, Chunk
except ImportError:
	from pyfastcdc.py import FastCDC, Chunk
	import warnings
	warnings.warn('Failed to import pyfastcdc cython extension, fallback to pure python implementation with is a lot slower.')

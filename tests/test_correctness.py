import hashlib
import io
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, List, NamedTuple

import pytest

from pyfastcdc.common import NormalizedChunking
from pyfastcdc.cy import FastCDC as FastCDC_cy
from pyfastcdc.py import FastCDC as FastCDC_py
from tests.utils import FastCDCType


class TestSekienAkashitaImage:
	class Param(NamedTuple):
		avg_size: int
		seed: int = 0
		nc: NormalizedChunking = 1
	# Param -> [(gear_hash, length), ...]
	EXPECTED_RESULT: Dict[Param, List[Tuple[int, int]]] = {}

	@pytest.mark.parametrize('cut_func', ['buf', 'stream'])
	@pytest.mark.parametrize('case_param', EXPECTED_RESULT.keys())
	def test_sekien_akashita(self, fastcdc_impl: FastCDCType, sekien_akashita_bytes: bytes, case_param: Param, cut_func: str):
		expected = self.EXPECTED_RESULT[case_param]

		cdc = fastcdc_impl(avg_size=case_param.avg_size, seed=case_param.seed, normalized_chunking=case_param.nc)

		if cut_func == 'buf':
			chunk_gen = cdc.cut_stream(BytesIO(sekien_akashita_bytes))
		elif cut_func == 'stream':
			chunk_gen = cdc.cut_buf(sekien_akashita_bytes)
		else:
			raise ValueError(cut_func)

		h = hashlib.sha256()
		chunk_cnt = 0
		for i, chunk in enumerate(chunk_gen):
			assert i < len(expected)
			assert chunk.gear_hash == expected[i][0]
			assert chunk.length == expected[i][1]
			h.update(chunk.data)
			chunk_cnt += 1
		assert chunk_cnt == len(expected)
		assert h.hexdigest() == hashlib.sha256(sekien_akashita_bytes).hexdigest()


class TestPyCyConsistency:
	DATA_SIZES = [
		*range(100),
		*[i * 100 for i in range(1, 10)],
		*[i * 1024 for i in range(1, 10)],
		*[i * 10 * 1024 for i in range(1, 5)],
		*[i * 100 * 1024 for i in range(1, 4)],
		*[i * 1024 * 1024 for i in range(1, 3)],
	]
	AVG_SIZES = [1024, 4096, 8192, 12345, 16384, 65536]

	@pytest.fixture(scope="class")
	def random_data_by_size(self) -> Dict[int, bytes]:
		data_cache = {}
		for size in self.DATA_SIZES:
			rnd = random.Random(size)
			data_cache[size] = bytes(rnd.getrandbits(8) for _ in range(size))
		return data_cache

	@pytest.mark.parametrize('avg_size', AVG_SIZES)
	@pytest.mark.parametrize('data_size', DATA_SIZES)
	@pytest.mark.parametrize('normalized_chunking', [0, 1, 2, 3])
	@pytest.mark.parametrize('seed', [0, 1])
	def test_py_cy_consistency(self, avg_size: int, data_size: int, normalized_chunking: NormalizedChunking, seed: int, random_data_by_size: Dict[int, bytes]):
		data = random_data_by_size[data_size]

		cdc_py = FastCDC_py(avg_size=avg_size, normalized_chunking=normalized_chunking, seed=seed)
		cdc_cy = FastCDC_cy(avg_size=avg_size, normalized_chunking=normalized_chunking, seed=seed)
		assert cdc_py.avg_size == cdc_cy.avg_size
		assert cdc_py.min_size == cdc_cy.min_size
		assert cdc_py.max_size == cdc_cy.max_size

		chunks_py = list(cdc_py.cut_buf(data))
		chunks_cy = list(cdc_cy.cut_buf(data))

		assert len(chunks_py) == len(chunks_cy)

		for py_chunk, cy_chunk in zip(chunks_py, chunks_cy):
			assert py_chunk.offset == cy_chunk.offset
			assert py_chunk.length == cy_chunk.length
			assert py_chunk.gear_hash == cy_chunk.gear_hash
			assert bytes(py_chunk.data) == bytes(cy_chunk.data)

		reconstructed_py = b''.join(bytes(chunk.data) for chunk in chunks_py)
		reconstructed_cy = b''.join(bytes(chunk.data) for chunk in chunks_cy)
		assert reconstructed_py == data
		assert reconstructed_cy == data


class TestFastCdcRsCompatibility:
	def test_non_power_of_two_avg_size_matches_fastcdc_rs_rounded_logarithm(self, fastcdc_impl: FastCDCType):
		# Given: deterministic data with an average size where floor(log2) and round(log2) diverge.
		rnd = random.Random(12288)
		data = bytes(rnd.getrandbits(8) for _ in range(200_000))
		expected_chunks = [
			(0, 15234, 16917778323220202162),
			(15234, 40563, 18147259289147603895),
			(55797, 12333, 6080427449276455385),
			(68130, 5634, 18267247979084204320),
			(73764, 13242, 16006360459130764060),
			(87006, 15295, 9837832051073236719),
			(102301, 37126, 10548008158650469598),
			(139427, 14118, 5452250018595482436),
			(153545, 18758, 4728851600197142336),
			(172303, 27697, 1685423970502062803),
		]

		# When: chunking with the non-power-of-two average size.
		cdc = fastcdc_impl(avg_size=12288)
		chunks = [(chunk.offset, chunk.length, chunk.gear_hash) for chunk in cdc.cut_buf(data)]

		# Then: output matches the rounded-logarithm mask selection used by fastcdc-rs v2020.
		assert chunks == expected_chunks


class TestCutMethods:
	def test_chunk_properties(self, fastcdc_instance, random_data_1m: bytes):
		prev_offset = None
		prev_size = None
		for chunk in fastcdc_instance.cut_buf(random_data_1m):
			assert isinstance(chunk.gear_hash, int)
			assert isinstance(chunk.offset, int)
			assert isinstance(chunk.length, int)
			assert isinstance(chunk.data, memoryview)

			assert len(chunk.data) == chunk.length

			if prev_offset is not None:
				assert chunk.offset == prev_offset + prev_size
			prev_offset = chunk.offset
			prev_size = chunk.length

	def test_cut_buf_vs_cut_file_consistency(self, fastcdc_impl: FastCDCType, random_data_1m: bytes, tmp_path: Path):
		cdc = fastcdc_impl(avg_size=8192)
		chunks_memory = list(cdc.cut_buf(random_data_1m))

		temp_file = tmp_path / 'test.bin'
		temp_file.write_bytes(random_data_1m)

		chunk_cnt = 0
		for i, chunk in enumerate(cdc.cut_file(temp_file)):
			chunk_cnt += 1
			assert chunk_cnt <= len(chunks_memory)
			assert chunk.offset == chunks_memory[i].offset
			assert chunk.length == chunks_memory[i].length
			assert chunk.gear_hash == chunks_memory[i].gear_hash
			assert bytes(chunk.data) == bytes(chunks_memory[i].data)
		assert len(chunks_memory) == chunk_cnt

	def test_cut_buf_vs_cut_stream_consistency_read(self, fastcdc_impl: FastCDCType, random_data_1m: bytes):
		cdc = fastcdc_impl(avg_size=8192)
		chunks_memory = list(cdc.cut_buf(random_data_1m))
		bytes_io = io.BytesIO(random_data_1m)

		class MyStream:
			def read(self, n: int) -> bytes:
				return bytes_io.read(n)

		chunk_cnt = 0
		for i, chunk in enumerate(cdc.cut_stream(MyStream())):
			chunk_cnt += 1
			assert chunk_cnt <= len(chunks_memory)
			assert chunk.offset == chunks_memory[i].offset
			assert chunk.length == chunks_memory[i].length
			assert chunk.gear_hash == chunks_memory[i].gear_hash
			assert bytes(chunk.data) == bytes(chunks_memory[i].data)
		assert len(chunks_memory) == chunk_cnt

	def test_cut_buf_vs_cut_stream_consistency_readinto(self, fastcdc_impl: FastCDCType, random_data_1m: bytes):
		cdc = fastcdc_impl(avg_size=8192)
		chunks_memory = list(cdc.cut_buf(random_data_1m))
		bytes_io = io.BytesIO(random_data_1m)

		class MyStream:
			def readinto(self, buf) -> int:
				return bytes_io.readinto(buf)

		chunk_cnt = 0
		for i, chunk in enumerate(cdc.cut_stream(MyStream())):
			chunk_cnt += 1
			assert chunk_cnt <= len(chunks_memory)
			assert chunk.offset == chunks_memory[i].offset
			assert chunk.length == chunks_memory[i].length
			assert chunk.gear_hash == chunks_memory[i].gear_hash
			assert bytes(chunk.data) == bytes(chunks_memory[i].data)
		assert len(chunks_memory) == chunk_cnt


class TestSeed:
	def test_different_seeds_produce_different_chunks(self, random_data_1m: bytes):
		cdc1 = FastCDC_cy(avg_size=8192, seed=1)
		cdc2 = FastCDC_cy(avg_size=8192, seed=2)

		chunks1 = list(cdc1.cut_buf(random_data_1m))
		chunks2 = list(cdc2.cut_buf(random_data_1m))

		hashes1 = [chunk.gear_hash for chunk in chunks1]
		hashes2 = [chunk.gear_hash for chunk in chunks2]

		assert hashes1 != hashes2 or len(chunks1) == 0


class TestChunkProperties:
	def test_chunk_data_integrity(self, fastcdc_instance, random_data_1m: bytes):
		data_list = []
		for chunk in fastcdc_instance.cut_buf(random_data_1m):
			data_list.append(bytes(chunk.data))
		assert b''.join(data_list) == random_data_1m

	def test_chunk_size_constraints(self, fastcdc_impl: FastCDCType, random_data_1m: bytes):
		avg_size = 16384
		min_size = avg_size // 4  # 4096
		max_size = avg_size * 4  # 65536

		cdc = fastcdc_impl(avg_size=avg_size)
		chunks = list(cdc.cut_buf(random_data_1m))

		for chunk in chunks:
			if chunk != chunks[-1]:
				assert min_size <= chunk.length <= max_size



def __build_test_data():
	cases = TestSekienAkashitaImage.EXPECTED_RESULT
	key = TestSekienAkashitaImage.Param
	cases[key(16384, 0)] = [
		(17968276318003433923, 21325),
		(8197189939299398838, 17140),
		(13019990849178155730, 28084),
		(4509236223063678303, 18217),
		(2504464741100432583, 24700),
	]
	cases[key(16384, 555, 0)] = [
		(11912081672558759016, 65536),
		(4429013657701329409, 15481),
		(4617353815122521848, 25471),
		(0, 2978),
	]
	cases[key(16384, 666, 1)] = [
        (9312357714466240148, 10605),
        (226910853333574584, 55745),
        (12271755243986371352, 11346),
        (14153975939352546047, 5883),
        (5890158701071314778, 11586),
        (8981594897574481255, 14301),
	]
	cases[key(16384, 777, 2)] = [
		(5756766315125948951, 35163),
		(17831515366867633926, 16422),
		(3216207887329694204, 17470),
		(1561632489531769228, 10294),
		(10369060306646295092, 18786),
		(14655496504067886486, 11331),
	]
	cases[key(16384, 888, 3)] = [
		(3158156329559295338, 17507),
		(5701878486563355731, 18589),
		(4529013630687100674, 17996),
		(3219799010358125664, 17281),
		(9266722211030107244, 15987),
		(7683217499559119532, 19292),
		(0, 2814),
	]
	cases[key(32768, 0)] = [
		(15733367461443853673, 66549),
		(6321136627705800457, 42917),
	]
	cases[key(65536, 0)] = [
		(2504464741100432583, 109466),
	]
	cases[key(1000, 0)] = [
		(15141981586395774332, 577),
		(8477613194938993865, 2025),
		(2735959075179562048, 1118),
		(18115765646671285894, 1458),
		(9500110725479242474, 1122),
		(443122261039895162, 334),
		(393511086818658384, 268),
		(16955226389877987730, 2234),
		(2735959075179562048, 1118),
		(18115765646671285894, 1458),
		(13954488060845043336, 616),
		(4626797161746623835, 4000),
		(11465206466110030982, 1108),
		(3877185904066068587, 1457),
		(5881250651602209508, 1321),
		(16925411787380621381, 251),
		(17489808562297316230, 860),
		(18129948957563258366, 1158),
		(13002932706949647172, 1369),
		(3370452418750950046, 258),
		(17998343902007936034, 1641),
		(11508990849237690054, 819),
		(5207468236900142596, 1090),
		(8284690672242526937, 1257),
		(16861457263863958963, 1179),
		(10960864681938687892, 1386),
		(6054483394976617000, 1068),
		(2386775949661439840, 1224),
		(9765031068850768480, 1037),
		(16719281564143894808, 1313),
		(17393396865315244987, 2021),
		(8197189939299398838, 320),
		(6871126333188003048, 885),
		(16972716804835771241, 1179),
		(12808154220016437278, 2139),
		(15385422482022415630, 356),
		(6844292880670166480, 1776),
		(13355204841389267792, 1168),
		(9374847859024726178, 971),
		(17084607867749926034, 1142),
		(10103105113398742526, 1278),
		(6254902090653532395, 2043),
		(7952073558027502250, 1534),
		(9616506964461227050, 1115),
		(12322483109039221194, 636),
		(8958286662040306431, 1341),
		(12554149114625989198, 1740),
		(16207690376253020852, 1006),
		(16248939738969512408, 1468),
		(14270957677345163169, 1299),
		(1551239710381196832, 1429),
		(14312347838531863880, 2478),
		(1335322050285636652, 1786),
		(1944458194743591353, 2275),
		(12039817357202303956, 949),
		(11820880714810263869, 1377),
		(16009206469796846404, 371),
		(14283166369001529626, 337),
		(15911323414792572684, 1076),
		(16417310421356693358, 868),
		(18147769296913861510, 1011),
		(15334853639705628667, 1769),
		(7045320843275616092, 482),
		(3743371431493807673, 665),
		(975175087488798292, 2264),
		(9731900868009236632, 1116),
		(7899942747347358014, 1264),
		(478520673225540621, 1025),
		(4509236223063678303, 683),
		(7911353879014065725, 1033),
		(17211447357795894654, 1597),
		(9814968862893591415, 1711),
		(6643495946427635752, 1356),
		(14635965274794233990, 1456),
		(17951524945264971610, 1006),
		(12407786606147719096, 1663),
		(8379360585841640134, 1234),
		(4798022593009727916, 291),
		(10955710641692053612, 1094),
		(457762767568428136, 344),
		(82832907034692228, 1444),
		(5774103846722045964, 1168),
		(17563493800282002670, 1450),
		(2724044535410791590, 1050),
		(9389354643814908759, 1085),
		(9132161234032204682, 1848),
		(16231621261041204310, 254),
		(261151940358997096, 1069),
		(9613379839789125245, 1323),
		(16119826487342861060, 1088),
		(0, 136),
	]
	cases[key(4096, 123)] = [
		(17640604251071308688, 5626),
		(17640604251071308688, 6534),
		(1164779717113740270, 5375),
		(1630317330713120546, 2204),
		(15925855439796383684, 7352),
		(14038325076263464756, 4459),
		(10372641652373760866, 2757),
		(6632718265737316290, 4345),
		(13547989595704043725, 13255),
		(5085413808508269073, 6871),
		(2088831145520463002, 1867),
		(17282638805110740002, 7908),
		(10392127899000403174, 4664),
		(12682705380171701888, 3262),
		(14163264652508939164, 4595),
		(11317546923098927019, 10371),
		(5442637979354971713, 5365),
		(918741369407724947, 3523),
		(11012502959421423036, 2886),
		(16443206251033749990, 4470),
		(18389377892984539132, 1777),
	]
	cases[key(3333, 456)] = [
		(10244635572917292382, 5352),
		(10244635572917292382, 6534),
		(6506656886334461048, 8428),
		(1568981245086988673, 4055),
		(976165388776206262, 8246),
		(227474149701075941, 1705),
		(15597163692867778936, 8896),
		(8853802354309906299, 1703),
		(7846154730496643533, 2469),
		(12579192060780867954, 876),
		(16367212629893111190, 6710),
		(11084210301959466778, 6617),
		(17216215298576800968, 2290),
		(2745509010730917144, 3990),
		(11949739281024563184, 2157),
		(11802567455667178519, 7639),
		(15836350909598670374, 3458),
		(3598944074651431428, 4450),
		(14136883845736820776, 6320),
		(3381712722571947258, 8934),
		(1258510376350532739, 1315),
		(1125375003867751948, 3341),
		(6560578492574969505, 3981),
	]
	cases[key(7696, 789)] = [
		(11334448432782874778, 8194),
		(12589329716764299352, 20178),
		(9249278391062916766, 8282),
		(17112004270336100794, 7728),
		(10241186054488990483, 11587),
		(13131687877381749692, 7247),
		(4691067104701485054, 13922),
		(14573090550801024034, 10300),
		(4483911677009026884, 3884),
		(16657773305248393930, 2912),
		(11098559764229607138, 8020),
		(15951496705909269620, 7212),
	]


__build_test_data()

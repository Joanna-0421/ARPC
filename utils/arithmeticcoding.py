# 
# Reference arithmetic coding
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 
import numpy as np
import sys
python3 = sys.version_info.major >= 3
import time

# ---- Arithmetic coding core classes ----

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase(object):
	
	# Constructs an arithmetic coder, which initializes the code range.
	def __init__(self, statesize):
#		if statesize < 1:
#			raise ValueError("State size out of range")
		
		# -- Configuration fields --
		# Number of bits for the 'low' and 'high' state variables. Must be at least 1.
		# - Larger values are generally better - they allow a larger maximum frequency total (MAX_TOTAL),
		#   and they reduce the approximation error inherent in adapting fractions to integers;
		#   both effects reduce the data encoding loss and asymptotically approach the efficiency
		#   of arithmetic coding using exact fractions.
		# - But larger state sizes increase the computation time for integer arithmetic,
		#   and compression gains beyond ~30 bits essentially zero in real-world applications.
		# - Python has native bigint arithmetic, so there is no upper limit to the state size.
		#   For Java and C++ where using native machine-sized integers makes the most sense,
		#   they have a recommended value of STATE_SIZE=32 as the most versatile setting.
		self.STATE_SIZE = statesize
		# Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
		self.MAX_RANGE = 1 << self.STATE_SIZE
		# Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
		self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
		# Maximum allowed total from a frequency table at all times during coding. This differs from Java
		# and C++ because Python's native bigint avoids constraining the size of intermediate computations.
		self.MAX_TOTAL = self.MIN_RANGE
		# Bit mask of STATE_SIZE ones, which is 0111...111.
		self.MASK = self.MAX_RANGE - 1
		# The top bit at width STATE_SIZE, which is 0100...000.
		self.TOP_MASK = self.MAX_RANGE >> 1
		# The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.
		self.SECOND_MASK = self.TOP_MASK >> 1
		
		# -- State fields --
		# Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
		self.low = 0
		# High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
		self.high = self.MASK
	
	
	# Updates the code range (low and high) of this arithmetic coder as a result
	# of processing the given symbol with the given frequency table.
	# Invariants that are true before and after encoding/decoding each symbol:
	# - 0 <= low <= code <= high < 2^STATE_SIZE. ('code' exists only in the decoder.)
	#   Therefore these variables are unsigned integers of STATE_SIZE bits.
	# - (low < 1/2 * 2^STATE_SIZE) && (high >= 1/2 * 2^STATE_SIZE).
	#   In other words, they are in different halves of the full range.
	# - (low < 1/4 * 2^STATE_SIZE) || (high >= 3/4 * 2^STATE_SIZE).
	#   In other words, they are not both in the middle two quarters.
	# - Let range = high - low + 1, then MAX_RANGE/4 < MIN_RANGE <= range
	#   <= MAX_RANGE = 2^STATE_SIZE. These invariants for 'range' essentially
	#   dictate the maximum total that the incoming frequency table can have.
	def update(self,  cumul, symbol):
		# State check
		#s = time.time()
		low = self.low
		high = self.high
#		if low >= high or (low & self.MASK) != low or (high & self.MASK) != high:
#			raise AssertionError("Low or high out of range")
		range = high - low + 1
#		if not (self.MIN_RANGE <= range <= self.MAX_RANGE):
#			raise AssertionError("Range out of range")
			
		# Frequency table values check
		total = cumul[-1].item()
		symlow = cumul[symbol].item()
		symhigh = cumul[symbol+1].item()
#		if symlow == symhigh:
#			raise ValueError("Symbol has zero frequency")
#		if total > self.MAX_TOTAL:
#			raise ValueError("Cannot code symbol because total is too large")
		
		# Update range
		newlow  = low + symlow  * range // total
		newhigh = low + symhigh * range // total - 1
		self.low = newlow
		self.high = newhigh
		# While the highest bits are equal
		#s1 = time.time()
		#print("update1", s1-s)
		while ((self.low ^ self.high) & self.TOP_MASK) == 0:
			self.shift()
			self.low = (self.low << 1) & self.MASK
			self.high = ((self.high << 1) & self.MASK) | 1
		
		# While the second highest bit of low is 1 and the second highest bit of high is 0
		#s2 = time.time()
		#print("update2", s2-s1)
		while (self.low & ~self.high & self.SECOND_MASK) != 0:
			self.underflow()
			self.low = (self.low << 1) & (self.MASK >> 1)
			self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1
	
		#s3 = time.time()
		#print("update3", s3-s2)
	
	# Called to handle the situation when the top bit of 'low' and 'high' are equal.
	def shift(self):
		raise NotImplementedError()
	
	
	# Called to handle the situation when low=01(...) and high=10(...).
	def underflow(self):
		raise NotImplementedError()



# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):
	
	# Constructs an arithmetic coding encoder based on the given bit output stream.
	def __init__(self, statesize, bitout):
		super(ArithmeticEncoder, self).__init__(statesize)
		# The underlying bit output stream.
		self.output = bitout
		# Number of saved underflow bits. This value can grow without bound.
		self.num_underflow = 0
	
	
	# Encodes the given symbol based on the given frequency table.
	# This updates this arithmetic coder's state and may write out some bits.
	def write(self, cumul, symbol):
#		if not isinstance(freqs, CheckedFrequencyTable):
#			freqs = CheckedFrequencyTable(freqs)
                #s = time.time()
                self.update(cumul, symbol)
                #print('update', time.time()-s)
	
	
	# Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
	# It is important that this method must be called at the end of the each encoding process.
	# Note that this method merely writes data to the underlying output stream but does not close it.
	def finish(self):
		#s = time.time()
		self.output.write(1)
		#print('finish', time.time()-s)
	
	
	def shift(self):
		#s = time.time()
		bit = self.low >> (self.STATE_SIZE - 1)
		self.output.write(bit)
		
		# Write out the saved underflow bits
                
		#s1 = time.time()
		#print('shift1', s1-s)
		for _ in range(self.num_underflow):
			self.output.write(bit ^ 1)
		self.num_underflow = 0
		#print('shift2', time.time()-s1)
	
	
	def underflow(self):
		self.num_underflow += 1



# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):
	
	# Constructs an arithmetic coding decoder based on the
	# given bit input stream, and fills the code bits.
	def __init__(self, statesize, bitin):
		super(ArithmeticDecoder, self).__init__(statesize)
		# The underlying bit input stream.
		self.input = bitin
		# The current raw code bits being buffered, which is always in the range [low, high].
		self.code = 0
		for _ in range(self.STATE_SIZE):
			self.code = self.code << 1 | self.read_code_bit()
	
	
	# Decodes the next symbol based on the given frequency table and returns it.
	# Also updates this arithmetic coder's state and may read in some bits.
	def read(self, cumul, alphabet_size):
#		if not isinstance(freqs, CheckedFrequencyTable):
#			freqs = CheckedFrequencyTable(freqs)
		
		# Translate from coding range scale to frequency table scale
		total = cumul[-1].item()
#		if total > self.MAX_TOTAL:
#			raise ValueError("Cannot decode symbol because total is too large")
		range = self.high - self.low + 1
		offset = self.code - self.low
		value = ((offset + 1) * total - 1) // range
#		assert value * range // total <= offset
#		assert 0 <= value < total
		
		# A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
		start = 0
		end = alphabet_size
		while end - start > 1:
			middle = (start + end) >> 1
			if cumul[middle] > value:
				end = middle
			else:
				start = middle
#		assert start + 1 == end
		
		symbol = start
#		assert freqs.get_low(symbol) * range // total <= offset < freqs.get_high(symbol) * range // total
		self.update(cumul, symbol)
#		if not (self.low <= self.code <= self.high):
#			raise AssertionError("Code out of range")
		return symbol
	
	
	def shift(self):
		self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()
	
	
	def underflow(self):
		self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()
	
	
	# Returns the next bit (0 or 1) from the input stream. The end
	# of stream is treated as an infinite number of trailing zeros.
	def read_code_bit(self):
		temp = self.input.read()
		if temp == -1:
			temp = 0
		return temp


class BitCollector:
    """
    一个伪输出流，它不写入文件，而是将所有比特收集到一个列表中。
    """
    def __init__(self):
        self.bits = []

    def write(self, bit):
        # 每当编码器尝试“写入”一个比特时，我们将其添加到列表中。
        self.bits.append(bit)

    def get_bits(self):
        # 一个方便的方法来获取最终的比特列表。
        return self.bits
    
class BitListReader:
	"""
	一个伪输入流，它从一个比特列表而不是文件中读取数据。
	"""
	def __init__(self, bits_list: list):
		self.bits = bits_list
		self.index = 0  # 追踪当前读取到哪个位置

	def read(self):
		# 检查是否已经读完列表中的所有比特
		if self.index >= len(self.bits):
			return -1  # 表示流的末尾 (End of Stream)
		
		# 获取当前位置的比特
		bit = self.bits[self.index]
		# 将索引向后移动一位
		self.index += 1
		return bit

	def get_consumed_bits_count(self):
		"""返回已经从流中读取的比特总数。"""
		return self.index

class BitOutputStream:
    """A simple bit-level output stream."""
    def __init__(self, file):
        self.file = file
        self.buffer = 0
        self.bit_count = 0

    def write(self, bit):
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self.buffer = (self.buffer << 1) | bit
        self.bit_count += 1
        if self.bit_count == 8:
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.bit_count = 0

    def flush(self):
        """
        将缓冲区中剩余的比特写入文件 (用0填充到整字节)，但不关闭文件。
        这是解决问题的关键。
        """
        if self.bit_count > 0:
            # 用0向左移位，填充剩余的比特位
            self.buffer <<= (8 - self.bit_count)
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.bit_count = 0

    def close(self):
        """刷新缓冲区并关闭底层文件。"""
        self.flush()
        self.file.close()

class BitInputStream:
    """A simple bit-level input stream."""
    def __init__(self, file):
        self.file = file
        self.buffer = 0
        self.bit_count = 0

    def read(self):
        if self.bit_count == 0:
            byte_data = self.file.read(1)
            if not byte_data:
                return -1  # End of stream
            self.buffer = byte_data[0]
            self.bit_count = 8
        
        # Read the most significant bit from the buffer
        bit = (self.buffer >> (self.bit_count - 1)) & 1
        self.bit_count -= 1
        return bit
        
    def close(self):
        self.file.close()

def probabilities_to_cumul(p0, total_freq=2**16):
    """
    将符号0的概率转换为算术编码器所需的累积频率表。
    
    Args:
        p0 (float): 符号0出现的概率 (0.0 to 1.0).
        total_freq (int): 用于模拟概率的整数总频率。

    Returns:
        numpy.ndarray: 累积频率表 (cumul)。
    """
    # 确保概率在有效范围内
    p0 = max(0.0, min(1.0, p0))
    
    # 计算频率，并确保每个符号的频率至少为1
    freq_0 = max(1, int(round(p0 * total_freq)))
    freq_1 = max(1, int(round((1.0 - p0) * total_freq)))
    
    # 重新计算总和
    current_total = freq_0 + freq_1
    
    # 创建累积频率表 [0, freq_0, freq_0 + freq_1]
    cumul = np.array([0, freq_0, current_total], dtype=np.uint64)
    return cumul

def compress_to_bit_list(bit_source, probabilities_of_zero):
    """
    将源比特流压缩成一个精确的比特列表。
    """
    bit_collector = BitCollector()
    encoder = ArithmeticEncoder(statesize=32, bitout=bit_collector)

    for i, bit in enumerate(bit_source):
        p0 = probabilities_of_zero[i]
        cumul = probabilities_to_cumul(p0) # probabilities_to_cumul 来自之前的答案
        encoder.write(cumul, bit)

    encoder.finish()
    
    return bit_collector.get_bits()


def decompress_from_bit_list(exact_bit_list, num_bits, probabilities_of_zero):
    """
    从一个精确的比特列表中解压缩出原始符号。
    """
    # 使用 BitListReader 将我们的列表包装成解码器可以使用的输入流
    bit_reader = BitListReader(exact_bit_list)
    decoder = ArithmeticDecoder(statesize=32, bitin=bit_reader)
    
    decoded_bits = []
    alphabet_size = 2
    
    for i in range(num_bits):
        # 解码时必须使用与编码时完全相同的概率序列
        p0 = probabilities_of_zero[i]
        cumul = probabilities_to_cumul(p0)
        
        bit = decoder.read(cumul, alphabet_size)
        decoded_bits.append(bit)
        
    return decoded_bits
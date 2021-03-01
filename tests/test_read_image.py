import unittest
import ocrolib
import PIL
import numpy

### Original image in disk and memory used to test method read_image_gray()
img_disk = "tests/testpage.png"
img_mem = PIL.Image.open(img_disk)

### Binarized image in disk and memory used to test method read_image_binary()
img_bin_disk = "tests/010030.bin.png"
img_bin_mem = PIL.Image.open(img_bin_disk)

class OcrolibTestCase(unittest.TestCase):
	"""
	Tests for processing image from disk and memory for methods 
	read_image_gray() and read_image_binary() in common.py under ocrolib
	"""

	def test_read_image_gray(self):
		"""
		Test whether the function read_image_gray() will return same result 
		when pass a image file name (from disk) and a image object (PIL.Image from memory).
		The return object of read_image_gray() is a 'ndarray' dedfined by 'numpy', thus we use the 
		built-in function 'array_equal' to compare two ndarray objects
		"""
		self.assertTrue(numpy.array_equal(ocrolib.read_image_gray(img_disk), ocrolib.read_image_gray(img_mem)))
		

	def test_read_image_binary(self):
		self.assertTrue(numpy.array_equal(ocrolib.read_image_binary(img_bin_disk), ocrolib.read_image_binary(img_bin_mem)))


if __name__ == '__main__':
	unittest.main()

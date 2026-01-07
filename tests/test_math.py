from utils.math_utils import (
  Round2,
  Clip1,
  Clip3,
  CeilLog2,
  FloorLog2,
  Round2Signed,
  bits_signed,
  integer_div,
  Round2,
  Round2Signed,
  bits_signed,
  integer_div,
  Round2,
  Round2Signed,
  bits_signed,
  integer_div,
)

def test_integer_div():
    assert integer_div(7, 4) == 1
    assert integer_div(-7, -4) == 1
    assert integer_div(-7, 4) == -1
    assert integer_div(7, -4) == -1

def test_Round2():
    assert Round2(0b1000, 3) == 0b1
    assert Round2(0b0100, 3) == 0b1
    assert Round2(0b1011, 3) == 0b1
    assert Round2(0b0111, 3) == 0b1

def test_FloorLog2():
    assert FloorLog2(0b1000) == 3
    assert FloorLog2(0b0100) == 2
    assert FloorLog2(0b1011) == 3
    assert FloorLog2(0b0111) == 2

def test_CeilLog2():
    assert CeilLog2(0b1000) == 3
    assert CeilLog2(0b0100) == 2
    assert CeilLog2(0b1011) == 4
    assert CeilLog2(0b1001) == 4

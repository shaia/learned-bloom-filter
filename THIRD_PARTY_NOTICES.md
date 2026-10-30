# Third-party notices

## FastPLBF

The construction algorithms in `include/plbf/optimize.hpp` and
`include/plbf/row_maxima.hpp` are a C++ port of the Python reference
implementation at <https://github.com/atsukisato/FastPLBF>.
`test/data/example.csv` is copied from that repository, and
`test/data/synth_*.csv` were produced by its synthetic-data generators
(`src/test.py`) through `tools/gen_golden.py`.

```
MIT License

Copyright (c) 2023 atsukisato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MurmurHash64A

`plbf::hash_bytes` in `include/plbf/hash.hpp` is MurmurHash64A by Austin
Appleby, which is in the public domain.

## Split-block Bloom filter layout

The block layout and salt constants in `include/plbf/split_block.hpp` follow
the Apache Parquet specification for split-block Bloom filters.

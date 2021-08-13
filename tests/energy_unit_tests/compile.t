package.terrapath = package.terrapath .. ';../../API/src/?.t'
local filename = arg[1]
print(filename)
-- _thallo_verbosity = 2
_thallo_threads_per_block = 256
_thallo_timing_level = 1
_thallo_double_precision = false
_thallo_verbosity = 2
local thallo = require("thallo")
thallo.dimensions = {[0] = 32,32,32,32,32,32,32,32,32}
thallo.problemkind = "gauss_newton"
local tbl = thallo.problemSpecFromFile(filename)
local result = thallo.compilePlan(tbl,thallo.problemkind)
print(result)
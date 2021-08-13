local cu = require("_unit_test_helpers")
local terra kernel(result : &float)
    var t = cu.linearThreadId()
    var peers = cu.get_peers([int](t) % 4)
    printf("%d: 0x%08x\n",cu.linearThreadId(), peers)
    cu.reduce_peersf(result + (t%4), [float](t), peers)
end
local R = terralib.cudacompile({ kernel = kernel },true)
local launcher = cu.wrap_float_warp_kernel(R.kernel)
local results = {}
for i=0,3 do results[i] = launcher(i) end
for i=0,3 do 
	print(tostring(i)..": "..tostring(results[i]))
	assert(results[i] == 112 + 8*i)
end
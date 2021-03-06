local cu = require("_unit_test_helpers")
local terra ballot_kernel(result : &int)
    var t = cu.linearThreadId()
    t = cu.ballot([int](t))
    cu.printf("%d: 0x%08x\n",cu.linearThreadId(), t)
    terralib.asm(terralib.types.unit,"red.global.max.u32 [$0], $1;","l,r",true,result,t)
end
local R = terralib.cudacompile({ ballot_kernel = ballot_kernel },true)
local launcher = cu.wrap_reduction_kernel(R.ballot_kernel)
local result = launcher()
assert(result == -2)
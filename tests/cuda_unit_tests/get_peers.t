local cu = require("_unit_test_helpers")
local terra kernel(result : &int)
    var t = cu.linearThreadId()
    t = cu.get_peers([int](t) % 4)
    printf("%d: 0x%08x\n",cu.linearThreadId(), t)
    t = t and 0x000000FF
    terralib.asm(terralib.types.unit,"red.global.add.u32 [$0], $1;","l,r",true,result,t)
end
local R = terralib.cudacompile({ kernel = kernel },true)
local launcher = cu.wrap_reduction_kernel(R.kernel)
local result = launcher()
assert(result == 255*32/4)
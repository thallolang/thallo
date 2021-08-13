package.terrapath = package.terrapath .. ';../../API/src/?.t'
if not terralib.cudacompile then
    print("CUDA not enabled, not performing test...")
    return
end
local cu = require("cuda_util")
local C = cu.C

function cu.wrap_reduction_kernel(kernel)
    return terra()
        var N = 32
        var d_data : &int
        C.cudaMalloc([&&opaque](&d_data),sizeof(int))
        C.cudaMemset(d_data,0,sizeof(int))
        var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
        kernel(&launch,d_data)
        var result = -1
        C.cudaMemcpy(&result,d_data,sizeof(int),2)
        return result
    end
end

function cu.wrap_float_warp_kernel(kernel)
    return terra(location : int32)
        var N = 32
        var d_data : &float
        var size = sizeof(float)*N
        C.cudaMalloc([&&opaque](&d_data),size)
        C.cudaMemset(d_data,0,size)
        var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
        kernel(&launch,d_data)
        var result : float[32]
        for i=0,32 do result[i] = 0 end
        C.cudaMemcpy(&result,d_data,size,2)
        return result[location]
    end
end


return cu
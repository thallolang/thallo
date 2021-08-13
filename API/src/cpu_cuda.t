local cu = {}

local ffi = require("ffi")
local S = require("std")
local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef _WIN32
#include <sys/time.h>
double CurrentTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}
#else
#include <Windows.h>
double CurrentTimeInSeconds() {
    unsigned __int64 pf;
    QueryPerformanceFrequency( (LARGE_INTEGER *)&pf );
    double freq_ = 1.0 / (double)pf;

    unsigned __int64 val;
    QueryPerformanceCounter( (LARGE_INTEGER *)&val );
    return (val) * freq_;
}
#endif
]]


cu.C = C

local warpSize = 32
cu.warpSize = warpSize

print("WARNING: USING CPU SIMULATION MODE OF THALLO. THIS IS A SLOW PROTOTYPE. IF POSSIBLE, DEVELOP USING THE DEFAULT GPU MODE ON A CUDA-ENABLED GPU")
function cu.loadCUDALibrary(libname,headername,successcode)
    error("loadCUDALibrary not available in CPU cuda mode")
    return nil
end

local GPUBlockDims = {"blockIdx","gridDim","threadIdx","blockDim"}
for i,d in ipairs(GPUBlockDims) do
    local tbl = {}
    for i,v in ipairs {"x","y","z" } do
        tbl[v] = global(0,d.."_"..v)
    end
    _G[d] = tbl
end

local cd = macro(function(apicall) 
    local loc = apicall.tree.filename..":"..apicall.tree.linenumber
    local apicallstr = tostring(apicall)
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda CPU shim reported error %d\n",r)
            C.printf("In call: %s", str)
            C.printf("From: %s\n", loc)
            C.exit(r)
        end
    in
        r
    end end)
cu.cd = cd

cu.checkedLaunch = macro(function(kernelName, apicall)
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var name = [kernelName]
        var r = apicall
        if r ~= 0 then  
            C.printf("Kernel %s, Cuda CPU shim reported error %d: %s\n", name, r)
            C.exit(r)
        end
    in
        r
    end end)

cu.printf = C.printf

terra cu.laneid()
    return cu.linearThreadId() % 32
end

terra cu.linearThreadId()
    var blockId = blockIdx.x 
             + blockIdx.y * gridDim.x 
             + gridDim.x * gridDim.y * blockIdx.z 
    return blockId * (blockDim.x * blockDim.y * blockDim.z)
              + (threadIdx.z * (blockDim.x * blockDim.y))
              + (threadIdx.y * blockDim.x)
              + threadIdx.x
end

cu.pascalOrBetterGPU = false

local type_to_ptx_char = {float = "f", int32 = "r", uint32 = "r"}
function cu.shfl(typ)
    return terra(v : typ, source_lane : uint)
        return v
    end
end


terra cu.atomicAddf(sum : &float, value : float)
    sum[0] = sum[0] + value
end

terra cu.__shfl_downf(v : float, delta : uint, width : int)
    return v
end

terra cu.__shfl_downd(v : double, delta : uint, width : int)
    return v;
end

terra cu.atomicAddd(sum : &double, value : double)
    sum[0] = sum[0] + value
end

-- https://devblogs.nvidia.com/parallelforall/voting-and-shuffling-optimize-atomic-operations/
terra cu.get_peers(key : int) : uint
    return (1 << cu.laneid())
end

terra cu.reduce_peersf(dest : &float, x : float, peers : uint)
    var lane : int = cu.laneid()
    if peers == (1 << lane) then
        cu.atomicAddf(dest, x)
    else
        C.printf("Invalid peer reduction for CPU, peers set to %u but lane mask is %u\n", lane, peers)
    end
end

terra cu.reduce_peersd(dest : &double, x : double, peers : uint)
    var lane : int = cu.laneid()
    if peers == (1 << lane) then
        cu.atomicAddd(dest, x)
    else
        C.printf("Invalid peer reduction for CPU, peers set to %u but lane mask is %u\n", lane, peers)
    end
end

-- Using the "Kepler Shuffle", see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
terra cu.warpReducef(val : float) 

  var offset = warpSize >> 1
  while offset > 0 do 
    val = val + cu.__shfl_downf(val, offset, warpSize);
    offset =  offset >> 1
  end
-- Is unrolling worth it?
  return val;
end

terra cu.warpReduced(val : double) 
  var offset = warpSize >> 1
  while offset > 0 do 
    val = val + cu.__shfl_downd(val, offset, warpSize);
    offset =  offset >> 1
  end
-- Is unrolling worth it?
  return val;
end


local terra cudaGetLastError()
    return 0
end
C.cudaGetLastError = cudaGetLastError

cu.maximumResidentThreadsPerGrid = 9223372036854775807

struct TimerEvent {
    time : double
}

C.cudaEvent_t = &TimerEvent

local terra cudaEventDestroy(event : C.cudaEvent_t);
    C.free(event)
end
C.cudaEventDestroy = cudaEventDestroy

local terra cudaEventCreate(event : &C.cudaEvent_t)
    @event = [C.cudaEvent_t](C.malloc(sizeof(TimerEvent)))
end
C.cudaEventCreate = cudaEventCreate

C.cudaStream_t = &int32

local terra cudaEventRecord(event : C.cudaEvent_t, stream : C.cudaStream_t)
    event.time = 0.0
    event.time = C.CurrentTimeInSeconds() * 1000.0
end
C.cudaEventRecord = cudaEventRecord

local terra cudaDeviceSynchronize() 
    return 0
end
C.cudaDeviceSynchronize = cudaDeviceSynchronize

local terra cudaEventSynchronize(event : C.cudaEvent_t) 
    return 0
end
C.cudaEventSynchronize = cudaEventSynchronize

local terra cudaEventElapsedTime(duration : &float, startEvent : C.cudaEvent_t, endEvent : C.cudaEvent_t)
    @duration = [float](endEvent.time - startEvent.time)
    return 0
end
C.cudaEventElapsedTime = cudaEventElapsedTime


local terra cudaMemcpy(dest : &opaque, src : &opaque, size : uint64, ignore : int)
    C.memcpy(dest,src,size)
    return 0
end
C.cudaMemcpy = cudaMemcpy

local terra cudaMemcpyAsync(dest : &opaque, src : &opaque, size : uint64, ignore : int, stream : C.cudaStream_t)
    C.memcpy(dest,src,size)
    return 0
end
C.cudaMemcpyAsync = cudaMemcpyAsync

local terra cudaMemsetAsync(dest : &opaque, val : int, count : uint64, stream : C.cudaStream_t)
    C.memset(dest,val,count)
    return 0
end
C.cudaMemsetAsync = cudaMemsetAsync

local terra cudaFree(ptr : &opaque)
    C.free(ptr)
    return 0
end
C.cudaFree = cudaFree

local terra cudaMalloc(dest : &&opaque, size : uint64)
    @dest = C.malloc(size)
    return 0
end
C.cudaMalloc = cudaMalloc

C.cudaMemcpyDeviceToHost = 0
C.cudaMemcpyDeviceToDevice = 0


local CUDAParams = terralib.types.newstruct("CUDAParams")
CUDAParams.entries = { { "gridDimX", uint },
                                { "gridDimY", uint },
                                { "gridDimZ", uint },
                                { "blockDimX", uint },
                                { "blockDimY", uint },
                                { "blockDimZ", uint },
                                { "sharedMemBytes", uint },
                                { "hStream" , terralib.types.pointer(opaque) } }
cu.CUDAParams = CUDAParams

function cu.cudacompile(module,dumpmodule)
    local m = {}
    for k,v in pairs(module) do
        if type(v) == "table" and terralib.isfunction(v.kernel) then
            v = v.kernel
        end
        local ktype = v:gettype()
        local symbols = ktype.parameters:map(terralib.newsymbol)
        m[k] = terra(params : &cu.CUDAParams, [symbols])
            [blockDim.x] = params.blockDimX
            [blockDim.y] = params.blockDimY
            [blockDim.z] = params.blockDimZ
            -- We get a compilation error when using the globals directly in the loop
            for bz = 0,params.gridDimZ do
                [blockIdx.z] = bz
                for by = 0,params.gridDimY do
                    [blockIdx.y] = by
                    for bx = 0,params.gridDimX do
                        [blockIdx.x] = bx
                        for tz = 0,params.blockDimZ do
                            [threadIdx.z] = tz
                            for ty = 0,params.blockDimY do
                                [threadIdx.y] = ty
                                for tx = 0,params.blockDimX do
                                    [threadIdx.x] = tx
                                    v([symbols])
                                end
                            end
                        end
                    end
                end
            end
            return 0
        end
    end
    return m
end

cu.global_memory = 64*10e9-- Placeholder 64GB

return cu
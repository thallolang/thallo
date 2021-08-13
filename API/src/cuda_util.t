local cu = {}

local ffi = require("ffi")

local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#ifdef _WIN32
    #include <io.h>
#endif

// Wrapper since terra cannot import static functions
cudaOccError cudaOccupancyCalculator(
    cudaOccResult               *result,           // out
    const cudaOccDeviceProp     *properties,       // in
    const cudaOccFuncAttributes *attributes,       // in
    const cudaOccDeviceState    *state,            // in
    int                          blockSize,        // in
    size_t                       dynamicSmemSize){ // in
    return cudaOccMaxActiveBlocksPerMultiprocessor(result, properties, attributes, state, blockSize, dynamicSmemSize);
}

]]

cu.C = C

local warpSize = 32
cu.warpSize = warpSize


local function generateCallMacro(cu_library, success_name, get_error_string_name)
    local callMacro = macro(function(apicall) 
        local loc = apicall.tree.filename..":"..apicall.tree.linenumber
        local apicallstr = tostring(apicall)
        local error_str = symbol(rawstring,"error_str")
        return quote
            var str = [apicallstr]
            var r = apicall
            if r ~= [cu_library[success_name]] then
                var [error_str] = "<unknown>"
                escape
                    if get_error_string_name then
                        emit quote error_str = [cu_library[get_error_string_name]](r) end
                    end
                end
                C.printf("Cuda Library reported error %d: %s\n", r, error_str) -- TODO write cublasGetErrorString
                C.printf("In call: %s", str)
                C.printf("From: %s\n", loc)
                C.exit(r)
            end
        in
            r
        end end)
    return callMacro
end

function cu.loadCUDALibrary(libname,headername,successcode, get_error_string_name)
    local cudapath = "/usr/local/cuda"
    local cudalibpath = "/lib64/lib"..libname..".dylib"
    if ffi.os == "Linux" then
        cudalibpath = "/lib64/lib"..libname..".so"
    end
    terralib.linklibrary(cudapath..cudalibpath)
    terralib.includepath = terralib.includepath..";"..
                           cudapath.."/include/"
    local incstring = [[
        #include <]]..headername..[[>
    ]]
    print(incstring)
    local library = terralib.includecstring(incstring)
    return library, generateCallMacro(library, successcode,get_error_string_name)
end

local libdevice = "/nvvm/libdevice/libdevice.10.bc"

local extern = terralib.externfunction
if terralib.linkllvm then
    local obj = terralib.linkllvm(libdevice)
    function extern(...) return obj:extern(...) end
else
    terralib.linklibrary(libdevice)
end
cu.extern = extern

local GPUBlockDims = {{"blockIdx","ctaid"},
              {"gridDim","nctaid"},
              {"threadIdx","tid"},
              {"blockDim","ntid"}}
for i,d in ipairs(GPUBlockDims) do
    local a,b = unpack(d)
    local tbl = {}
    for i,v in ipairs {"x","y","z" } do
        local fn = cudalib["nvvm_read_ptx_sreg_"..b.."_"..v] 
        tbl[v] = `fn()
    end
    _G[a] = tbl
end

local cd = macro(function(apicall) 
    local loc = apicall.tree.filename..":"..apicall.tree.linenumber
    local apicallstr = tostring(apicall)
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
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
            C.printf("Kernel %s, Cuda reported error %d: %s\n", name, r, C.cudaGetErrorString(r))
            C.exit(r)
        end
    in
        r
    end end)

local vprintfname = "cudart:vprintf"
local vprintf = terralib.externfunction(vprintfname, {&int8,&int8} -> int)

local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    for i,e in ipairs(args) do
        local typ = e:gettype()
        local field = "_"..tonumber(i)
        typ = typ == float and double or typ
        table.insert(Buf.entries,{field,typ})
    end
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                emit quote
                   buf.["_"..tonumber(i)] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end

cu.printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)

terra cu.laneid()
    var laneid : int
    laneid = terralib.asm(int,"mov.u32 $0, %laneid;","=r", true)
    return laneid
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

cu.__syncthreads = cudalib.nvvm_barrier0

terra cu.atomicAddI32(sum : &int32, value : int32)
    return terralib.asm(int32,"atom.global.add.s32 $0,[$1],$2;","=r,l,r", true, sum, value)
end

cu.ffs = extern("__nv_ffs", int -> int)
cu.popc = extern("__nv_popc", uint -> int)

-- This call implicitly links in the cuda runtime if it has not yet been loaded
cu.pascalOrBetterGPU = cudalib.localversion() >= 60

--[[
__device__ unsigned int __ballot(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.ballot.b32 \t%0, %%p1; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}
--]]
terra cu.ballot(pred : int) : uint
    var ret : uint32
    ret = terralib.asm(uint32,"{\n\t.reg .pred %ballot_p; \n\tsetp.ne.u32 %ballot_p, $1, 0; \n\tvote.ballot.b32 $0, %ballot_p; \n\t}","=r,r",true,pred);
    return ret
end

terra cu.any(pred : int) : bool
    -- TODO: implement using intrinsics...
    --printf("any(%d)\n",[int](pred))
    var b = cu.ballot(pred)
    --printf("any(%u),%d\n",b,[int](pred))
    return b ~= 0
    --[[
    var iPred = [int](pred)
    var ret : uint32
    ret = terralib.asm(uint32,"{\n\t.reg .pred %any_p; \n\tsetp.ne.u32 %any_p, $1, 0; \n\tvote.any.b32 $0, %any_p; \n\t}","=r,r",true,iPred);
    return [bool](ret)
    --]]
end


struct ULLDouble {
    union {
        a : uint64;
        b : double;
    }
}

struct uint2 {
    x : uint32;
    y : uint32;
}

struct uint2Double {
    union {
        u2 : uint2;
        d: double;
    }
}

local terra __double_as_ull(v : double)
    var u : ULLDouble
    u.b = v;

    return u.a;
end

local terra __ull_as_double(v : uint64)
    var u : ULLDouble
    u.a = v;

    return u.b;
end



local type_to_ptx_char = {float = "f", int32 = "r", uint32 = "r"}
function cu.shfl(typ)
    if typ == double then
        local ch = type_to_ptx_char["uint32"]
        return terra (v : typ, source_lane : uint)
            var ret : uint2Double;
            var init : uint2Double;
            init.d = v
            var c : int =  0x1F
            ret.u2.x = terralib.asm(uint32,"shfl.idx.b32 $0, $1, $2, $3;",["="..ch..","..ch..",r,r"], true, init.u2.x, source_lane, c)
            ret.u2.y = terralib.asm(uint32,"shfl.idx.b32 $0, $1, $2, $3;",["="..ch..","..ch..",r,r"], true, init.u2.y, source_lane, c)
            return ret.d
        end
    else
        local ch = type_to_ptx_char[tostring(int32)]

        assert(ch, "Invalid shfl type: "..tostring(typ))
        return terra (v : typ, source_lane : uint)
            var ret : typ
            var c : int =  0x1F
            ret = terralib.asm(typ,"shfl.idx.b32 $0, $1, $2, $3;",["="..ch..","..ch..",r,r"], true, v, source_lane, c)
            return ret
        end
    end
end


terra cu.atomicAddf(sum : &float, value : float)
    terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
end

terra cu.__shfl_downf(v : float, delta : uint, width : int)
    var ret : float;
    var c : int;
    c = ((warpSize-width) << 8) or 0x1F;
    ret = terralib.asm(float,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, v, delta, c)
    return ret;
end

terra cu.__shfl_downd(v : double, delta : uint, width : int)
    var ret : uint2Double;
    var init : uint2Double;
    init.d = v
    var c : int;
    c = ((warpSize-width) << 8) or 0x1F;
    ret.u2.x = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.x, delta, c)
    ret.u2.y = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.y, delta, c)
    return ret.d;
end

if cu.pascalOrBetterGPU then
    terra cu.atomicAddd(sum : &double, value : double)
        var address_as_i : uint64 = [uint64] (sum);
        terralib.asm(terralib.types.unit,"red.global.add.f64 [$0],$1;","l,d", true, address_as_i, value)
    end
else
    terra cu.atomicAddd(sum : &double, value : double)
        var address_as_i : &uint64 = [&uint64] (sum);
        var old : uint64 = address_as_i[0];
        var assumed : uint64;

        repeat
            assumed = old;
            old = terralib.asm(uint64,"atom.global.cas.b64 $0,[$1],$2,$3;", 
                "=l,l,l,l", true, address_as_i, assumed, 
                __double_as_ull( value + __ull_as_double(assumed) )
                );
        until assumed == old;

        return __ull_as_double(old);
    end
end

-- https://devblogs.nvidia.com/parallelforall/voting-and-shuffling-optimize-atomic-operations/
terra cu.get_peers(key : int) : uint
    var peers : uint = 0
    var is_peer : bool = false 

    -- in the beginning, all lanes are available
    var unclaimed : uint32 = 0xffffffff

    while not is_peer do 
        -- fetch key of first unclaimed lane and compare with this key
        is_peer = (key == [cu.shfl(int)](key, cu.ffs(unclaimed) - 1))

        -- determine which lanes had a match
        peers = cu.ballot([int](is_peer))

        -- remove lanes with matching keys from the pool
        unclaimed = unclaimed ^ peers

        -- quit if we had a match
    end

    return peers
end

terra cu.reduce_peersf(dest : &float, x : float, peers : uint)
    var lane : int = cu.laneid()

    -- find the peer with lowest lane index
    var first : int = cu.ffs(peers)-1;

    -- calculate own relative position among peers
    var rel_pos : int  = cu.popc(peers << (32 - lane));

    -- ignore peers with lower (or same) lane index
    peers = peers and (0xfffffffe << lane)
    while cu.any(peers) do
        -- find next-highest remaining peer
        var nxt = cu.ffs(peers);
        -- __shfl() only works if both threads participate, so we always do.
        var t : float = [cu.shfl(float)](x, nxt - 1)

        -- only add if there was anything to add
        if nxt > 0 then x = x + t end

        -- all lanes with their least significant index bit set are done
        var done : uint = rel_pos and 1

        -- remove all peers that are already done
        peers = peers and (not cu.ballot([int](done)))

        -- abuse relative position as iteration counter
        rel_pos = rel_pos >> 1
    end

    if dest ~= [&float](0) and lane == first then -- only leader threads for each key perform atomics
        --printf("lane: %d\n", lane)
        cu.atomicAddf(dest, x)
    end
end

terra cu.reduce_peersd(dest : &double, x : double, peers : uint)
    var lane : int = cu.laneid()

    -- find the peer with lowest lane index
    var first : int = cu.ffs(peers)-1;

    -- calculate own relative position among peers
    var rel_pos : int  = cu.popc(peers << (32 - lane));

    -- ignore peers with lower (or same) lane index
    peers = peers and (0xfffffffe << lane)
    while cu.any(peers) do
        -- find next-highest remaining peer
        var nxt = cu.ffs(peers);
        -- __shfl() only works if both threads participate, so we always do.
        var t : double = [cu.shfl(double)](x, nxt - 1)

        -- only add if there was anything to add
        if nxt > 0 then x = x + t end

        -- all lanes with their least significant index bit set are done
        var done : uint = rel_pos and 1

        -- remove all peers that are already done
        peers = peers and (not cu.ballot([int](done)))

        -- abuse relative position as iteration counter
        rel_pos = rel_pos >> 1
    end

    if dest ~= [&double](0) and lane == first then -- only leader threads for each key perform atomics
        --printf("lane: %d\n", lane)
        cu.atomicAddd(dest, x)
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



terra cu.terraMaximumResidentThreadsPerGrid()
    var device : int = 0
    cd(C.cudaGetDevice(&device))
    var deviceProp : C.cudaDeviceProp
    cd(C.cudaGetDeviceProperties(&deviceProp, device))
    var maxResidentThreadsPerSM = 2048
    if deviceProp.major == 2 then
        maxResidentThreadsPerSM = 1536
    end
    return deviceProp.multiProcessorCount * maxResidentThreadsPerSM
end


cu.maximumResidentThreadsPerGrid = cu.terraMaximumResidentThreadsPerGrid()

cu.CUDAParams = terralib.CUDAParams

cu.cudacompile = terralib.cudacompile


local terra toYesNo(pred : int32)
    if pred ~= 0 then return "Yes" else return  "No" end
end

local terra toEnDisabled(pred : int32)
    if pred ~= 0 then return "Enabled" else return  "Disabled" end
end

local terra printCudaDeviceProperties()
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)

    C.printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name)

    C.cudaDriverGetVersion(&driverVersion)
    C.cudaRuntimeGetVersion(&runtimeVersion)
    C.printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    C.printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor)

    C.printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
            [float](deviceProp.totalGlobalMem)/1048576.0f, [uint64](deviceProp.totalGlobalMem))
    -- TODO: compute # "cores"
    C.printf("  (%2d) Multiprocessors\n",   deviceProp.multiProcessorCount)
    C.printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    C.printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    C.printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize ~= 0) then
        C.printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    end

    C.printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
           deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    C.printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
           deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    C.printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
           deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    C.printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    C.printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    C.printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    C.printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    C.printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    C.printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    C.printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    C.printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    C.printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
    C.printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
    C.printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", toYesNo(deviceProp.deviceOverlap), deviceProp.asyncEngineCount);
    C.printf("  Run time limit on kernels:                     %s\n", toYesNo(deviceProp.kernelExecTimeoutEnabled))
    C.printf("  Integrated GPU sharing Host Memory:            %s\n", toYesNo(deviceProp.integrated))
    C.printf("  Support host page-locked memory mapping:       %s\n", toYesNo(deviceProp.canMapHostMemory))
    C.printf("  Alignment requirement for Surfaces:            %s\n", toYesNo(deviceProp.surfaceAlignment))
    C.printf("  Device has ECC support:                        %s\n", toEnDisabled(deviceProp.ECCEnabled))
    C.printf("  Device supports Unified Addressing (UVA):      %s\n", toYesNo(deviceProp.unifiedAddressing))
    C.printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
end


-- Adapted from  CUDA samples helper_cuda.h:
terra cores_per_SM(major : int32, minor : int32)
--[[
  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};
--]]

    if major == 3 then return 192 end
    if major == 5 then return 128 end
    if major == 6 and minor > 0 then return 128 end
    -- else
    return 64
end


terra global_memory()
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)
    return [uint64](deviceProp.totalGlobalMem)
end

-- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation
-- Return value is in bytes/second
terra theoretical_memory_bandwidth() : double
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)
    var mem_Hz = deviceProp.memoryClockRate * 1e3f
    var busWidthBytes = deviceProp.memoryBusWidth / 8.0
    return mem_Hz*busWidthBytes*2.0
end

terra theoretical_peak_flops() : double
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)
    var compute_Hz = deviceProp.clockRate * 1e3f
    var core_count = cores_per_SM(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount
    return compute_Hz*core_count*2.0 -- for FMA
end

terra core_count() : uint64
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)
    return cores_per_SM(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount
end

terra uncoalesced_multiplier() : int
    var transaction_size = 32
    var scalar_size = 4
    return transaction_size / scalar_size
end

terra transactions_per_coalesced_read() : int
    var warp_size = 32
    return warp_size / uncoalesced_multiplier()
end

cu.core_count                   = tonumber(core_count())
cu.global_memory                = tonumber(global_memory())
cu.theoretical_memory_bandwidth = theoretical_memory_bandwidth()
cu.theoretical_peak_flops       = theoretical_peak_flops()
cu.uncoalesced_multiplier       = uncoalesced_multiplier()
cu.transactions_per_coalesced_read = transactions_per_coalesced_read()

terra cu.get_max_active_threads(register_count : int32)
    var dev : int32
    var driverVersion = 0
    var runtimeVersion = 0;
    C.cudaGetDevice(&dev)
    var deviceProp : C.cudaDeviceProp
    C.cudaGetDeviceProperties(&deviceProp, dev)

    var result : C.cudaOccResult
    var prop : C.cudaOccDeviceProp
    prop.computeMajor                   = deviceProp.major                       -- Compute capability major version
    prop.computeMinor                   = deviceProp.minor                       -- Compute capability minor
    prop.maxThreadsPerBlock             = deviceProp.maxThreadsPerBlock          -- Maximum number of threads per block
    prop.maxThreadsPerMultiprocessor    = deviceProp.maxThreadsPerMultiProcessor -- Maximum number of threads per SM  i.e. (Max. number of warps) x (warp size)
    prop.regsPerBlock                   = deviceProp.regsPerBlock                -- Maximum number of registers per block
    prop.regsPerMultiprocessor          = deviceProp.regsPerMultiprocessor       -- Maximum number of registers per SM
    prop.warpSize                       = deviceProp.warpSize                    -- Warp size
    prop.sharedMemPerBlock              = deviceProp.sharedMemPerBlock           -- Maximum shared memory size per block
    prop.sharedMemPerMultiprocessor     = deviceProp.sharedMemPerMultiprocessor  -- Maximum shared memory size per SM
    prop.numSms                         = deviceProp.multiProcessorCount         -- Number of SMs available
    
    var attributes : C.cudaOccFuncAttributes
    C.memset(&attributes, 0, sizeof(C.cudaOccFuncAttributes))
    attributes.maxThreadsPerBlock =  prop.maxThreadsPerBlock
    attributes.numRegs = register_count          
    attributes.sharedSizeBytes = 0 -- Number of static shared memory used
    attributes.partitionedGCConfig = C.PARTITIONED_GC_OFF 

    var state : C.cudaOccDeviceState
    C.memset(&state, 0, sizeof(C.cudaOccDeviceState))
    
    var blockSize : int32 = _thallo_threads_per_block
    var dynamicSmemSize : C.size_t = 0
    var err_code = C.cudaOccupancyCalculator(&result, &prop, &attributes, &state,          
                            blockSize, dynamicSmemSize);
    if err_code ~= C.CUDA_OCC_SUCCESS then
        if err_code == C.CUDA_OCC_ERROR_INVALID_INPUT then
            C.printf("cudaOccMaxActiveBlocksPerMultiprocessor: INVALID_INPUT")
        elseif err_code == C.CUDA_OCC_ERROR_UNKNOWN_DEVICE then
            C.printf("cudaOccMaxActiveBlocksPerMultiprocessor: UNKNOWN_DEVICE")
        else 
            C.printf("cudaOccMaxActiveBlocksPerMultiprocessor: UNKNOWN ERROR")
        end
        C.exit(err_code)
    end
    return deviceProp.multiProcessorCount * result.activeBlocksPerMultiprocessor * _thallo_threads_per_block 
end


if _thallo_verbosity >= 1 then
    print("GPU Summary Information")
    print("Core Count: ", cu.core_count)
    print("Global Memory (bytes): ", cu.global_memory)
    print("Theoretical Peak Bandwidth (bytes/s): ", cu.theoretical_memory_bandwidth)
    print("Theoretical Peak Compute (FLOPs): ", cu.theoretical_peak_flops)
    print("Compute-to-Memory Amortization Ratio: ", cu.theoretical_peak_flops/cu.theoretical_memory_bandwidth)
end

if _thallo_verbosity > 1 then
    printCudaDeviceProperties()
end

return cu
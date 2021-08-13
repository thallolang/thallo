local S = require("std")
require("precision")
local A = require("ir")
require "fun" ()
local cu = nil
if _thallo_use_cpu_only==1 then
    cu = require("cpu_cuda")
else
    cu = require("cuda_util")
end
local util = {}

-- Turn on for consistent kernel names for debugging
local fixed_key = true

util.A = A
util.C = cu.C
util.cu = cu
local C = util.C
local verbosePTX    = _thallo_verbosity > 3
local verboseTrace  = _thallo_verbosity > 2

local extern    = cu.extern
local nativeDoubleAtomics = cu.pascalOrBetterGPU
util.warpSize   = cu.warpSize
util.laneid     = cu.laneid
util.printf = cu.printf
util.global_memory = cu.global_memory

if thallo_float == float then
    util.atomicAdd = cu.atomicAddf
    util.reduce_peers = cu.reduce_peersf
    util.warpReduce = cu.warpReducef
else
    util.atomicAdd = cu.atomicAddd
    util.reduce_peers = cu.reduce_peersd
    util.warpReduce = cu.warpReduced
end
if _thallo_use_cpu_only==1 then
    util.reduce = macro(function(reductionTarget,val) return quote
        util.atomicAdd(reductionTarget, val)
    end end)
else
    util.reduce = macro(function(reductionTarget,val) return quote
        val = util.warpReduce(val)
        if (util.laneid() == 0) then                
            util.atomicAdd(reductionTarget, val)
        end
    end end)
end

terra util.warp_aggregated_atomic_reduction_by_key(sum : &thallo_float, peers : uint, value : thallo_float)
    if peers == 0xffffffff then
        util.reduce(sum,value)
    else
        util.reduce_peers(sum, value, peers)
    end
end

local cd = cu.cd
util.cd = cd
local checkedLaunch = cu.checkedLaunch
util.get_peers = cu.get_peers


-- Must match Thallo.h
struct Thallo_PerformanceEntry {
    count : uint32 
    minMS : double
    maxMS : double
    meanMS : double
    stddevMS : double
}

struct Thallo_PerformanceSummary {
    -- Performance Statistics for full solves
    total               : Thallo_PerformanceEntry
    -- Performance Statistics for individual nonlinear iterations,
    -- This is broken up into three rough categories below
    nonlinearIteration  : Thallo_PerformanceEntry
    nonlinearSetup      : Thallo_PerformanceEntry
    linearSolve         : Thallo_PerformanceEntry
    nonlinearResolve    : Thallo_PerformanceEntry
}
util.Thallo_PerformanceEntry,util.Thallo_PerformanceSummary = Thallo_PerformanceEntry,Thallo_PerformanceSummary

function util.uniquify(lst)
    local existing = {}
    local result = terralib.newlist()
    for i,v in ipairs(lst) do
        if not existing[v] then 
            result:insert(v)
            existing[v] = true
        end
    end
    return result
end

--[[ rPrint(struct, [limit], [indent])   Recursively print arbitrary data. 
    Set limit (default 100) to stanch infinite loops.
    Indents tables as [KEY] VALUE, nested tables as [KEY] [KEY]...[KEY] VALUE
    Set indent ("") to prefix each line:    Mytable [KEY] [KEY]...[KEY] VALUE
--]]
function util.rPrint(s, l, i) -- recursive Print (structure, limit, indent)
    l = (l) or 100; i = i or "";    -- default item limit, indent string
    local ts = type(s);
    if (l<1) then print (i,ts," *snip* "); return end;
    if (ts ~= "table") then print (i,ts,s); return end
    print (i,ts);           -- print "table"
    for k,v in pairs(s) do  -- print "[KEY] VALUE"
        util.rPrint(v, l-1, i.."\t["..tostring(k).."]");
    end
end 

function util.allcomponentsofclass(index,class)
    return all(function(c) return class:isclassof(c) end, index.components)
end

table.indexOf = function( t, object )
    if "table" == type( t ) then
        for i = 1, #t do
            if object == t[i] then
                return i
            end
        end
        return -1
    else
        error("table.indexOf expects table for first argument, " .. type(t) .. " given")
    end
end

if (_thallo_use_cpu_only == 0) and thallo_float == double and (not nativeDoubleAtomics) then
    print("Warning: double precision on GPUs with compute capability < 6.0 (before Pascal) have no native double precision atomics, so we must use slow software emulation instead. This has a large performance impact on graph energies.")
end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function util.filter(L,fn)
    local newL = terralib.newlist()
    for _,v in ipairs(L) do
        if fn(v) then newL:insert(v) end
    end
    return newL
end

local mathParamCount = {sqrt = 1,
cos  = 1,
acos = 1,
sin  = 1,
asin = 1,
tan  = 1,
atan = 1,
ceil = 1,
floor = 1,
log = 1,
exp = 1,
pow  = 2,
fmod = 2,
fmax = 2,
fmin = 2
}


util.gpuMath = {}
util.cpuMath = {}
for k,v in pairs(mathParamCount) do
	local params = {}
	for i = 1,v do
		params[i] = thallo_float
	end
    suffix = ""
    if (thallo_float == float) then
        suffix = "f"
    end
    util.cpuMath[k] = C[k..suffix]
    if _thallo_use_cpu_only==1 then
        util.gpuMath[k] = util.cpuMath[k]
    else
        util.gpuMath[k] = extern(("__nv_%s"..suffix):format(k), params -> thallo_float)
    end
end
if (thallo_float == float) then
    util.cpuMath["abs"] = C["fabsf"]
else
    util.cpuMath["abs"] = C["fabs"]
end
util.gpuMath["abs"] = terra (x : thallo_float)
	if x < 0 then
		x = -x
	end
	return x
end
util.gpuMath["abs"]:setname("abs")

local Vectors = {}
function util.isvectortype(t) return Vectors[t] end
util.Vector = terralib.memoize(function(typ,N)
    N = assert(tonumber(N),"expected a number")
    local ops = { "__sub","__add","__mul","__div" }
    local struct VecType { 
        data : typ[N]
    }
    Vectors[VecType] = true
    VecType.metamethods.type, VecType.metamethods.N = typ,N
    VecType.metamethods.__typename = function(self) return ("%s_%d"):format(tostring(self.metamethods.type),self.metamethods.N) end
    for i, op in ipairs(ops) do
        local i = symbol(int,"i")
        local function template(ae,be)
            return quote
                var c : VecType
                for [i] = 0,N do
                    c.data[i] = operator(op,ae,be)
                end
                return c
            end
        end
        local terra opvv(a : VecType, b : VecType) [template(`a.data[i],`b.data[i])]  end
        local terra opsv(a : typ, b : VecType) [template(`a,`b.data[i])]  end
        local terra opvs(a : VecType, b : typ) [template(`a.data[i],`b)]  end
        
        local doop
        if terralib.overloadedfunction then
            doop = terralib.overloadedfunction("doop",{opvv,opsv,opvs})
        else
            doop = opvv
            doop:adddefinition(opsv:getdefinitions()[1])
            doop:adddefinition(opvs:getdefinitions()[1])
        end
        
       VecType.metamethods[op] = doop
    end
    terra VecType.metamethods.__unm(self : VecType)
        var c : VecType
        for i = 0,N do
            c.data[i] = -self.data[i]
        end
        return c
    end
    terra VecType:abs()
       var c : VecType
       for i = 0,N do
	  -- TODO: use thallo.abs
	  if self.data[i] < 0 then
	     c.data[i] = -self.data[i]
	  else
	     c.data[i] = self.data[i]
	  end
       end
       return c
    end
    terra VecType:sum()
       var c : typ = 0
       for i = 0,N do
	  c = c + self.data[i]
       end
       return c
    end
    terra VecType:dot(b : VecType)
        var c : typ = 0
        for i = 0,N do
            c = c + self.data[i]*b.data[i]
        end
        return c
    end
    terra VecType:max()
        var c : typ = 0
        if N > 0 then
            c = self.data[0]
        end
        for i = 1,N do
            if self.data[i] > c then
                c = self.data[i]
            end
        end
        return c
    end
	terra VecType:size()
        return N
    end
    terra VecType.methods.FromConstant(x : typ)
        var c : VecType
        for i = 0,N do
            c.data[i] = x
        end
        return c
    end
    VecType.metamethods.__apply = macro(function(self,idx) return `self.data[idx] end)
    VecType.metamethods.__cast = function(from,to,exp)
        if from:isarithmetic() and to == VecType then
            return `VecType.FromConstant(exp)
        end
        error(("unknown vector conversion %s to %s"):format(tostring(from),tostring(to)))
    end
    return VecType
end)

function Array(T,debug)
    local struct Array(S.Object) {
        _data : &T;
        _size : int32;
        _capacity : int32;
    }
    function Array.metamethods.__typename() return ("Array(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Array:init() : &Array
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Array:reserve(cap : int32)
        if cap > 0 and cap > self._capacity then
            var oc = self._capacity
            if self._capacity == 0 then
                self._capacity = 16
            end
            while self._capacity < cap do
                self._capacity = self._capacity * 2
            end
            self._data = [&T](S.realloc(self._data,sizeof(T)*self._capacity))
        end
    end
    terra Array:initwithcapacity(cap : int32) : &Array
        self:init()
        self:reserve(cap)
        return self
    end
    terra Array:__destruct()
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        if self._data ~= nil then
            C.free(self._data)
            self._data = nil
        end
    end
    terra Array:size() return self._size end
    
    terra Array:get(i : int32)
        assert(i < self._size) 
        return &self._data[i]
    end
    Array.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Array:insertNatlocation(idx : int32, N : int32, v : T) : {}
        assert(idx <= self._size)
        self._size = self._size + N
        self:reserve(self._size)

        if self._size > N then
            var i = self._size
            while i > idx do
                self._data[i - 1] = self._data[i - 1 - N]
                i = i - 1
            end
        end

        for i = 0ULL,N do
            self._data[idx + i] = v
        end
    end
    terra Array:insertatlocation(idx : int32, v : T) : {}
        return self:insertNatlocation(idx,1,v)
    end
    terra Array:insert(v : T) : {}
        return self:insertNatlocation(self._size,1,v)
    end
    terra Array:remove(idx : int32) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end
    if not T:isstruct() then
        terra Array:indexof(v : T) : int32
            for i = 0LL,self._size do
                if (v == self._data[i]) then
                    return i
                end
            end
            return -1
        end
        terra Array:contains(v : T) : bool
            return self:indexof(v) >= 0
        end
    end
    return Array
end

local Array = S.memoize(Array)

terra stringIndex(arr : &Array(rawstring), str : rawstring)
    for i = 0LL,arr._size do
        if (C.strcmp(str,arr._data[i])) == 0 then
            return i
        end
    end
    return -1
end

util.symTable = function(typ, N, name)
	local r = terralib.newlist()
	for i = 1, N do
		r[i] = symbol(typ, name..tostring(i-1))
	end
	return r
end

util.ceilingDivide = terra(a : int32, b : int32)
	return (a + b - 1) / b
end

struct RunningStats {
    count : uint32
    sum : double
    sqSum : double
    min : double
    max : double
}
terra RunningStats:init(newVal : float)
    self.count = 1
    self.sum   = [double](newVal)
    self.sqSum = [double](newVal*newVal)
    self.min   = self.sum
    self.max   = self.sum
end
terra RunningStats:update(newVal : float)
    self.count = self.count + 1
    self.sum   = self.sum   + [double](newVal)
    self.sqSum = self.sqSum + [double](newVal*newVal)
    self.min   = C.fmin(self.min,[double](newVal))
    self.max   = C.fmax(self.max,[double](newVal))
end


util.TimerEvent = C.cudaEvent_t

struct util.TimingInfo {
    startEvent : util.TimerEvent
    endEvent : util.TimerEvent
    duration : float
    eventName : rawstring
}
local TimingInfo = util.TimingInfo

struct util.Timer {
    timingInfo : &Array(TimingInfo)
}
local Timer = util.Timer

terra Timer:init() 
    self.timingInfo = [Array(TimingInfo)].alloc():init()
end

terra Timer:cleanup()
    for i = 0,self.timingInfo:size() do
        var eventInfo = self.timingInfo(i);
        C.cudaEventDestroy(eventInfo.startEvent);
        C.cudaEventDestroy(eventInfo.endEvent);
    end
    self.timingInfo:delete()
end 


terra Timer:startEvent(name : rawstring,  stream : C.cudaStream_t, endEvent : &util.TimerEvent)
    var timingInfo : TimingInfo
    timingInfo.eventName = name
    if [_thallo_timing_level > 2] then cd(C.cudaDeviceSynchronize()) end
    C.cudaEventCreate(&timingInfo.startEvent)
    C.cudaEventCreate(&timingInfo.endEvent)
    C.cudaEventRecord(timingInfo.startEvent, stream)
    self.timingInfo:insert(timingInfo)
    @endEvent = timingInfo.endEvent
end
terra Timer:endEvent(stream : C.cudaStream_t, endEvent : util.TimerEvent)
    if [_thallo_timing_level > 2] then cd(C.cudaDeviceSynchronize()) end
    C.cudaEventRecord(endEvent, stream)
end

terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end
terra computeSummary(stats : RunningStats) : Thallo_PerformanceEntry
    var summary : Thallo_PerformanceEntry
    var N = stats.count
    var mean = stats.sum/N
    var moment2 = stats.sqSum/N
    var variance = moment2 - (mean*mean)
    var stddev = C.sqrt(C.fabs(variance))
    summary.count = N
    summary.meanMS = mean
    summary.minMS = stats.min
    summary.maxMS = stats.max
    summary.stddevMS = stddev
    return summary
end

local terra setSummaryEntry(name : rawstring, entry : &Thallo_PerformanceEntry, names : &Array(rawstring), stats : &Array(RunningStats))
    var idx : int32
    idx =  stringIndex(names,name)
    if idx >= 0 then @entry = computeSummary((@stats)(idx)) end
end

terra Timer:evaluate(perfSummary : &Thallo_PerformanceSummary)
    cd(C.cudaDeviceSynchronize())
	var runningStats = [Array(RunningStats)].salloc():init()
	var aggregateTimingNames = [Array(rawstring)].salloc():init()
	for i = 0,self.timingInfo:size() do
		var eventInfo = self.timingInfo(i);
		C.cudaEventSynchronize(eventInfo.endEvent)
    	C.cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
    	var index =  stringIndex(aggregateTimingNames,eventInfo.eventName)
        if isprefix("Linear It",eventInfo.eventName) then
            C.printf("Linear Iter %7.4f\n", eventInfo.duration)
        end
    	if index < 0 then
            aggregateTimingNames:insert(eventInfo.eventName)
            var s : RunningStats
            s:init(eventInfo.duration)
            runningStats:insert(s)
        else
    		runningStats(index):update(eventInfo.duration)
    	end
    end
    setSummaryEntry("Total", &perfSummary.total, aggregateTimingNames, runningStats)
    setSummaryEntry("Nonlinear Iteration", &perfSummary.nonlinearIteration, aggregateTimingNames, runningStats)
    setSummaryEntry("Nonlinear Setup", &perfSummary.nonlinearSetup, aggregateTimingNames, runningStats)
    setSummaryEntry("Linear Solve", &perfSummary.linearSolve, aggregateTimingNames, runningStats)
    setSummaryEntry("Nonlinear Finish", &perfSummary.nonlinearResolve, aggregateTimingNames, runningStats)

    if ([_thallo_verbosity > 0]) then
        -- Markdown-style table
        C.printf("\n")
	    C.printf(		"|        Kernel        |   Count  | Total(ms) | Average(ms) | Std. Dev(ms) |\n")
		C.printf(		"|----------------------|----------|-----------|-------------|--------------|\n")
	    for i = 0, aggregateTimingNames:size() do
            var sum =   runningStats(i).sum
            var sqSum = runningStats(i).sqSum
            var N =     runningStats(i).count
            var mean = sum/N
            var moment2 = sqSum/N
            var variance = moment2 - (mean*mean)
            var stddev = C.sqrtf(C.fabsf(variance))
            C.printf("| %-20s |   %4u   | %8.3f  |   %8.4f  |   %8.4f   |\n", aggregateTimingNames(i), 
                N, sum, mean, stddev) --(sqSum-(sum*sum))/N)
            
        end
        C.printf(       "|--------------------------------------------------------------------------|\n")
        C.printf("TIMING ")
        for i = 0, aggregateTimingNames:size() do
            var n = aggregateTimingNames(i)
            if isprefix("PCGInit1",n) or isprefix("PCGStep1",n) or isprefix("Total",n) then
                C.printf("%f ",runningStats(i).sum)
            end
        end
        C.printf("\n")
        -- TODO: Refactor timing code
        var linIters = 0
        var nonLinIters = 0
        for i = 0, aggregateTimingNames:size() do
            var n = aggregateTimingNames(i)
            if isprefix("PCGInit1",n) then
                linIters = runningStats(i).count
            end
            if isprefix("PCGStep1",n) then
                nonLinIters = runningStats(i).count
            end
        end
        var linAggregate : float = 0.0f
        var nonLinAggregate : float = 0.0f
        for i = 0, aggregateTimingNames:size() do
            var n = runningStats(i).count
            if n == linIters then
                linAggregate = linAggregate + runningStats(i).sum 
            end
            if n == nonLinIters then
                nonLinAggregate = nonLinAggregate + runningStats(i).sum 
            end
        end
        C.printf("Per-iter times ms (nonlinear,linear): %7.4f\t%7.4f\n", linAggregate, nonLinAggregate)
    end
    
end

util.max = terra(x : double, y : double)
    return terralib.select(x > y, x, y)
end

local function noHeader(pd)
    return quote end
end

local function noFooter(pd)
    return quote end
end

util.initParameters = function(self, ProblemSpec, params, isInit)
    local stmts = terralib.newlist()
    for _, entry in ipairs(ProblemSpec.parameters) do
        if entry.kind == "ImageParam" then
            if entry.idx ~= "alloc" then
                local function_name = isInit and "initFromGPUptr" or "setGPUptr"
                local loc = entry.isunknown and (`self.X.[entry.name]) or `self.[entry.name]
                stmts:insert quote
                    loc:[function_name]([&uint8](params[entry.idx]))
                end
                if verboseTrace then
                    stmts:insert quote
                        C.printf([entry.name..":"..function_name.."params("..tostring(entry.idx)..")\n"])
                    end
                end
            end
        else
            local rhs
            if entry.kind == "SparseParam" then
                rhs = `[&entry.outspace:indextype()](params[entry.idx])
            elseif entry.kind == "ScalarParam" and entry.idx >= 0 then
                rhs = `@[&entry.type](params[entry.idx])
            end
            if rhs then
                stmts:insert quote self.[entry.name] = rhs end
                if verboseTrace then
                    stmts:insert quote
                        C.printf([entry.name.." = params[???]\n"])
                    end
                end
            end
        end
	end
	return stmts
end

util.validateParameters = function(self, ProblemSpec, params)
    local stmts = terralib.newlist()
    stmts:insert quote C.printf(" === Validating Parameters Start\n") end
    for _, entry in ipairs(ProblemSpec.parameters) do
        if entry.kind == "ImageParam" then
            if entry.idx ~= "alloc" then
                local loc = entry.isunknown and (`self.X.[entry.name]) or `self.[entry.name]
                stmts:insert quote
                    do
                        var size = loc:totalbytes()
                        C.printf([entry.name..": Attempting to read all %llu bytes from device for parameter "..entry.idx..".\n"],size)
                        var data : &opaque
                        cd(C.cudaMalloc(&data,size))
                        cd(C.cudaMemcpy(data,params[entry.idx],size,C.cudaMemcpyDeviceToDevice))
                        C.printf("Success\n")
                        cd(C.cudaFree(data))
                    end
                end
            end
        else
            stmts:insert quote
                do
                    C.printf(["Validation for "..entry.kind.." NYI; skipping validation for "..entry.name..": at index "..entry.idx..".\n"])
                end
            end
        end
    end
    stmts:insert quote C.printf(" === Validating Parameters Complete\n") end
    return stmts
end


util.initPrecomputedImages = function(self, ProblemSpec)
    local stmts = terralib.newlist()
	for _, entry in ipairs(ProblemSpec.parameters) do
        if entry.kind == "ImageParam" and entry.idx == "alloc" then
            stmts:insert quote
    		    self.[entry.name]:initGPU()
    		end
    	end
    end
    return stmts
end

util.freePrecomputedImages = function(self, ProblemSpec)
    local stmts = terralib.newlist()
    for _, entry in ipairs(ProblemSpec.parameters) do
        if entry.kind == "ImageParam" and entry.idx == "alloc" then
            stmts:insert quote
                self.[entry.name]:freeData()
            end
        end
    end
    return stmts
end

local positionForValidLane = util.positionForValidLane

-- Assumes x is (nonnegative) power of 2
local function iLog2(x)
    local result = 0
    while x > 1 do
        result = result + 1
        x = x / 2
    end
    return result
end

-- Get the block dimensions that best approximate a square/cube 
-- in 2 or 3 dimensions while only using power of 2 side lengths.
local function getBlockDims(blockSize)
    local LOG_BLOCK_SIZE = iLog2(blockSize)
    local dim2x = math.ceil(LOG_BLOCK_SIZE/2)
    local dim2y = LOG_BLOCK_SIZE - dim2x

    local dim3x = math.ceil(LOG_BLOCK_SIZE/3)
    local dim3y = math.ceil((LOG_BLOCK_SIZE - dim3x)/2)
    local dim3z = LOG_BLOCK_SIZE - dim3x - dim3y

    return { {blockSize,1,1}, {math.pow(2,dim2x),math.pow(2,dim2y),1}, {math.pow(2,dim3x),math.pow(2,dim3y),math.pow(2,dim3z)} }
end

local BLOCK_SIZE = _thallo_threads_per_block
assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE should be a multiple of the warp size (32), but is "..tostring(BLOCK_SIZE))

local BLOCK_DIMS = getBlockDims(BLOCK_SIZE)

local THREADS_PER_UNKNOWN = 512
util.THREADS_PER_UNKNOWN = THREADS_PER_UNKNOWN

assert(THREADS_PER_UNKNOWN == 1 or THREADS_PER_UNKNOWN % 32 == 0)


local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel)
    kernelName = kernelName.."_"..tostring(ft)
    local kernelparams = compiledKernel:gettype().parameters
    local params = terralib.newlist {}
    for i = 3,#kernelparams do --skip GPU launcher and PlanData
        params:insert(symbol(kernelparams[i]))
    end
    print("makeGPULauncher "..kernelName.." "..tostring(ft))
    local function createLaunchParameters(pd)
        if ft.kind == "UnknownwiseFunction" then
            local ispace = ft.domain:IndexSpace()
            local exps = terralib.newlist()
            for i = 1,3 do
               local dim = #ispace.dims >= i and ispace.dims[i].size or 1
                local bs = BLOCK_DIMS[#ispace.dims][i]
                exps:insert(dim)
                exps:insert(bs)
            end
            return exps
        elseif ft.kind == "ResidualwiseFunction" or 
            ft.kind == "ResidualAndContractionwiseFunction" or 
            ft.kind == "GenericFunction" or 
            ft.kind == "IterationDomainwiseFunction" then
            local elementcount = ft:cardinality()
            return {`[elementcount],BLOCK_DIMS[1][1],1,1,1,1}
        end
        assert(false)
    end
    local terra GPULauncher(pd : &PlanData, [params])
        var xdim,xblock,ydim,yblock,zdim,zblock = [ createLaunchParameters(pd) ]
            
        var launch = cu.CUDAParams { (xdim - 1) / xblock + 1, (ydim - 1) / yblock + 1, (zdim - 1) / zblock + 1, 
                                            xblock, yblock, zblock, 
                                            0, nil }
        var stream : C.cudaStream_t = nil
        var endEvent : C.cudaEvent_t 
        if ([_thallo_timing_level] > 1) then
            pd.hd.timer:startEvent(kernelName,nil,&endEvent)
        end
        if [verboseTrace] then
            C.printf("Kernel %s called with params: {gridDim(%u,%u,%u), blockDim(%u,%u,%u)}\n",
                kernelName, launch.gridDimX, launch.gridDimY, launch.gridDimZ, launch.blockDimX, launch.blockDimY, launch.blockDimZ)
        end
        checkedLaunch(kernelName, compiledKernel(&launch, @pd, params))
        if [verboseTrace] then
            C.printf("Kernel called with params: {gridDim(%u,%u,%u), blockDim(%u,%u,%u)}\n",
                launch.gridDimX, launch.gridDimY, launch.gridDimZ, launch.blockDimX, launch.blockDimY, launch.blockDimZ)
            cd(C.cudaDeviceSynchronize())
        end
        
        if ([_thallo_timing_level] > 1) then
            pd.hd.timer:endEvent(nil,endEvent)
        end

        cd(C.cudaGetLastError())
    end
    return GPULauncher
end

function util.makeGPUFunctions(problemSpec, dimensions, PlanData, delegate, names)
    -- Put unknownwise functions first; when we have both unknownwise and 
    -- residualwise, unknownwise initializes output arrays
    local function functionOrder(f1,f2)
        local f1_u = f1.typ.kind == "UnknownwiseFunction"
        local f2_u = f2.typ.kind == "UnknownwiseFunction"
        return (f1_u and not f2_u)
    end
    problemSpec.functions:sort(functionOrder)


    -- step 1: compile the actual cuda kernels
    local kernelFunctions = {}
    local key = fixed_key and "" or tostring(os.time())
    local function getkname(name,ft,suffix)
        local mangled = string.gsub(tostring(ft), "[%(%)]", "_")
        local kname = string.format("%s_%s_%s",name,mangled,key)
        if suffix then
            kname = kname.."_"..tostring(suffix)
        end
        return kname
    end
    local function addUnknownwiseFunctionsWithIspace(ispace,fmap,postfix,i)
        local dimcount = #ispace.dims
        assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
        local ks = delegate.UnknownwiseFunctions(ispace,fmap)
        for name,func in pairs(ks) do
            kernelFunctions[getkname(name,postfix,i)] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[dimcount][1]}, {"maxntidy", BLOCK_DIMS[dimcount][2]}, {"maxntidz", BLOCK_DIMS[dimcount][3]}, {"minctasm",1} } }
        end
    end
    local ispacesWithExclude = {}
    for i,problemfunction in ipairs(problemSpec.functions) do
        local fmap = problemfunction.functionmap
        if problemfunction.typ.kind == "ResidualwiseFunction" 
            or problemfunction.typ.kind == "ResidualAndContractionwiseFunction" then
            local ks = delegate.ResidualwiseFunctions(problemfunction.typ,fmap)
            for name,func in pairs(ks) do
                kernelFunctions[getkname(name,problemfunction.typ,i)] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[1][1]}, {"minctasm",1} } }
            end
        elseif problemfunction.typ.kind == "IterationDomainwiseFunction" then
            local ks = delegate.IterationDomainwiseFunctions(problemfunction.typ,fmap)
            for name,func in pairs(ks) do
                kernelFunctions[getkname(name,problemfunction.typ,i)] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[1][1]}, {"minctasm",1} } }
            end
        elseif problemfunction.typ.kind == "UnknownwiseFunction" then
            local ispace = problemfunction.typ.domain:IndexSpace()
            ispacesWithExclude[ispace] = ispace
            addUnknownwiseFunctionsWithIspace(ispace,fmap,problemfunction.typ,i)
        else
            assert(false)
        end
    end
    for _,ispace in ipairs(problemSpec._UnknownType.ispaces) do
        if not ispacesWithExclude[ispace] then
            local fmap = {}
            fmap.exclude = macro(function() return `false end)
            addUnknownwiseFunctionsWithIspace(ispace,fmap,ispace,0)
        end
    end

    do
        local ks = delegate.FlatUnknownwiseFunctions()
        for name,func in pairs(ks) do
            kernelFunctions[getkname(name,problemSpec:FlatUnknownwiseFunction())] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[1][1]}, {"minctasm",1} } }
        end
        ks = delegate.JTJwiseFunctions()
        for name,func in pairs(ks) do
            kernelFunctions[getkname(name,problemSpec:JTJwiseFunction())] = { kernel = func , annotations = { {"maxntidx", BLOCK_DIMS[1][1]}, {"minctasm",1} } }
        end
    end
    
    local kernels = cu.cudacompile(kernelFunctions, verbosePTX)

    -- step 2: generate wrapper functions around each named thing
    local grouplaunchers = {}
    for _,name in ipairs(names) do
        local args
        local launches = terralib.newlist()
        for _,ispace in ipairs(problemSpec._UnknownType.ispaces) do
            if not ispacesWithExclude[ispace] then
                local kname = getkname(name,ispace,0)
                local kernel = kernels[kname]
                if kernel then
                    local launcher = makeGPULauncher(PlanData, name, A.UnknownwiseFunction(ispace:getiterationdomain()), kernel)
                    if not args then
                        args = launcher:gettype().parameters:map(symbol)
                    end
                    launches:insert(`launcher(args))
                end
            end
        end
        local genericFunctionTypes = terralib.newlist()
        genericFunctionTypes:insert(problemSpec:FlatUnknownwiseFunction())
        genericFunctionTypes:insert(problemSpec:JTJwiseFunction())
        for _,ft in ipairs(genericFunctionTypes) do
            local kname = getkname(name,ft)
            local kernel = kernels[kname]
            if kernel then
                print("Grabbing GenericFunction "..kname)
                local launcher = makeGPULauncher(PlanData, name, ft, kernel)
                if not args then
                    args = launcher:gettype().parameters:map(symbol)
                end
                launches:insert(`launcher(args))
            end
        end
        for i,problemfunction in ipairs(problemSpec.functions) do
            local kname = getkname(name,problemfunction.typ,i)
            local kernel = kernels[kname]
            if kernel then -- some domains do not have an associated kernel, (see _Finish kernels in GN which are only defined for 
                local launcher = makeGPULauncher(PlanData, name, problemfunction.typ, kernel)
                if not args then
                    args = launcher:gettype().parameters:map(symbol)
                end
                launches:insert(`launcher(args))
            else
                --print("not found: "..name.." for "..tostring(problemfunction.typ))
            end
        end
        local fn
        if not args then
            fn = macro(function() return `{} end) -- dummy function for blank groups occur for things like precompute when they are not present
        else
            fn = terra([args]) launches end
            fn:setname(name)
            fn:gettype()
        end
        grouplaunchers[name] = fn 
    end
    return grouplaunchers
end

local temp = terralib.includecstring [[
typedef union uf32
{
    unsigned u;
    float f;
} uf32;

int isnan( float value )
{
    uf32 ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) > 0x7f800000;
}

int isinf( float value )
{
    uf32 ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) == 0x7f800000;
}

int isfinite( float value )
{
    uf32 ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) < 0x7f800000;
}

]]
C.isnan    = temp.isnan
C.isinf    = temp.isinf
C.isfinite = temp.isfinite


function permgen (a, n)
  if n == 0 then
    coroutine.yield(a)
  else
    for i=1,n do

      -- put i-th element as the last one
      a[n], a[i] = a[i], a[n]

      -- generate all permutations of the other elements
      permgen(a, n - 1)

      -- restore i-th element
      a[n], a[i] = a[i], a[n]

    end
  end
end

function util.permutations(a)
  local n = table.getn(a)
  return coroutine.wrap(function () permgen(a, n) end)
end


local List = terralib.newlist

function util.cartesian_product(sets)
  local result = List()
  local set_count = #sets
  local yield = coroutine.yield 
  local function descend(depth)
    if depth == set_count then
      for _,v in ipairs(sets[depth]) do
        result[depth] = v
        local resultcopy = List()
        for i=1,depth do
            resultcopy[i] = result[i]
        end
        yield(resultcopy)
      end
    else
      for _,v in ipairs(sets[depth]) do
        result[depth] = v
        descend(depth + 1)
      end
    end
  end
  return coroutine.wrap(function() descend(1) end)
end

function util.powerset(s)
   local t = List()
   t:insert(List())
   for i = 1, #s do
      for j = 1, #t do
        local newsubset = List()
        newsubset:insert(s[i])
        newsubset:insertall(t[j])
        t:insert(newsubset)
      end
   end
   return t
end

function util.exists(lst,pred)
    for _,v in ipairs(lst) do
        if pred(v) then
            return true
        end
    end
    return false
end

function util.extract_domains(component)
    assert(A.IndexComponent:isclassof(component))
    local domains = terralib.newlist()
    local function extract_helper(current)
        if A.DirectIndexComponent:isclassof(current) then
            domains:insert(current.domain)
        elseif A.SparseIndexComponent:isclassof(current) then
            for _,c in ipairs(current.access.index.components) do
                extract_helper(c)
            end
        elseif A.ComponentBinOp:isclassof(current) then
            extract_helper(current.lhs)
            extract_helper(current.rhs)
        end
    end
    extract_helper(component)
    return util.uniquify(domains)
end

function util.extract_domain(component)
    local domains = util.extract_domains(component)
    assert(#domains == 1)
    return domains[1]
end

function util.extract_domains_from_index(index)
    assert(A.ImageIndex:isclassof(index))
    local list_of_domains = index.components:map(util.extract_domains)
    for i=2,#list_of_domains do
        list_of_domains[1]:insertall(list_of_domains[i])
    end
    return util.uniquify(list_of_domains[1])
end

function util.list_subtraction(set1,set2)
    local result = List()
    for _,v in ipairs(set1) do
        if not util.exists(set2,function(v2) return v2 == v end) then
            result:insert(v) 
        end
    end
    return result
end

function util.wrap(class)
    return function(d) return class(d) end
end

return util

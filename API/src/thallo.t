thallo = {} --anchor it in global namespace, otherwise it can be collected
local S = require("std")
local ffi = require("ffi")
local util = require("util")
local thallolib = require("lib")
ad = require("ad")
require("precision")
local A = require("ir")
require("pprofiler")
local autoscheduler = require("autoscheduler")
local List = terralib.newlist
local C = util.C
local Thallo_PerformanceSummary    = util.Thallo_PerformanceSummary
local Thallo_PerformanceEntry      = util.Thallo_PerformanceEntry

--[[ For debuggging only ]]--
local use_contiguous_allocation = true
local guard_atomics = true
local profile_plan = false

local limited_search_space = false
local deatomize = false

local stub_out_image_loads = false
local print_image_indices = false -- dependent on stub_out_image_loads

local stub_out_sparse_loads = false
local print_sparse_indices = false 

local stub_out_param_access = false
local print_param_access = false

local use_new_autoscheduler = true

-- constants
local verboseSolver = _thallo_verbosity > 0
local verboseAD     = _thallo_verbosity > 1
local verboseTexture = _thallo_verbosity > 1
local verboseTrace = _thallo_verbosity > 2
util.verboseTrace = verboseTrace

printf = util.printf
local dprint

local actualPrintf = macro(function(fmt,...)
    local args = {...}
    return `C.printf(fmt, args)
end)
local nullPrintf = macro(function(fmt,...)
    return 0
end)

logSolver = verboseSolver and actualPrintf or nullPrintf
logAD = verboseAD and actualPrintf or nullPrintf
logTexture = verboseTexture and actualPrintf or nullPrintf
logTrace = verboseTrace and actualPrintf or nullPrintf
if verboseAD then
    dprint = print
else
    dprint = function() end
end

local gauss_newton = require("gauss_newton")
local ffi = require('ffi')

-- [[ GLOBAL STATE! ]] --
local problems = {}
local index_domain_index = 0
local ComputedArrayCache = {}
local computed_array_count = 0
local active_problem_spec = nil

local function compilePlan(problemSpec, kind, dimensions)
    assert(kind == "gauss_newton" or kind == "levenberg_marquardt" ,"expected solver kind to be gauss_newton or levenberg_marquardt")
    return gauss_newton(problemSpec, dimensions)
end

struct thallo.Plan(S.Object) {
    init : {&opaque,&&opaque} -> {} -- plan.data,params
    setsolverparameter : {&opaque,rawstring,&opaque} -> {} -- plan.data,name,param
    getsolverparameter : {&opaque,rawstring,&opaque} -> {}
    free : {&opaque} -> {} -- plan.data
    step : {&opaque,&&opaque} -> int
    cost : {&opaque} -> double
    get_summary : {&opaque,&Thallo_PerformanceSummary} -> {}
    estimated_cost : {&opaque} -> double -- performance estimate
    reset_unknowns : {&opaque} -> {}
    data : &opaque
    problemid : int
    dimensions : &uint32
}

struct thallo.Problem {} -- just used as an opaque type, pointers are actually just the ID
local function problemDefine(filename, kind, pid)
    local problemmetadata = { filename = ffi.string(filename), kind = ffi.string(kind), id = #problems + 1 }
    problems[problemmetadata.id] = problemmetadata
    pid[0] = problemmetadata.id
end
problemDefine = terralib.cast({rawstring, rawstring, &int} -> {}, problemDefine)

local function problemDelete(pid)
    problems[pid] = "DELETED"
end
problemDelete = terralib.cast({int} -> {}, problemDelete)

thallo.PSpec = A.ProblemSpec
local PROBLEM_STAGES  = { inputs = 0, functions = 1 }
function thallo.ProblemSpec()
    local ps = A.ProblemSpec()
    ps.parameters = terralib.newlist() -- ProblemParam*
    ps.names = {} -- name -> index in parameters list
    ps.functions = List() -- ProblemFunctions*
    ps.maxStencil = 0
    ps.stage = "inputs"
    ps.usepreconditioner = false
    ps.problemkind = thallo.problemkind
    return ps
end

function A.ProblemSpec:UsePreconditioner(v)
    self:Stage "inputs"
    self.usepreconditioner = v
end
function A.ProblemSpec:Stage(name)
    assert(PROBLEM_STAGES[self.stage] <= PROBLEM_STAGES[name], "all inputs must be specified before functions are added")
    self.stage = name
end

function A.ProblemSpec:registername(name)
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
end

function A.ProblemParam:terratype() return self.type end
function A.ImageParam:terratype() return self.imagetype:terratype() end
function A.SparseParam:terratype() return &self.outspace:indextype() end
function A.ProblemSpec:MaxStencil()
    self:Stage "functions"
    return self.maxStencil
end

function A.ScheduledEnergy:JTJpSchedules()
    if not self._JTJpSchedules then
        self._JTJpSchedules = List()
        for i,rg in ipairs(self.residualgroups) do
            self._JTJpSchedules:insert(rg.domainandschedule.schedule.jtjpschedule)
        end
    end
    return self._JTJpSchedules
end


function A.ScheduledEnergy:CanFuseJtJpReduction()
    -- TODO: Make more robust
    for i,rg in ipairs(self.residualgroups) do
        if A.ResidualSchedule:isclassof(rg.domainandschedule.schedule.fnschedule) or 
                (rg.domainandschedule.schedule.jtjpschedule ~= A.INLINE) then
            return false
        end
    end
    return true
end

function A.ScheduledEnergy:CanFusePCGInit()
    -- TODO: Make more robust
    for i,rg in ipairs(self.residualgroups) do
        if A.ResidualSchedule:isclassof(rg.domainandschedule.schedule.jtfschedule) then
            return false
        end
    end
    return true
end
function A.ScheduledEnergy:ContainsAnyOfJTJpSchedules(types)
    local scheds = self:JTJpSchedules()
    local contains = util.exists(scheds,
        function(s)
            return util.exists(types, function(t) return t:isclassof(s) end)
        end)
    return contains
end
function A.ScheduledEnergy:RequiresJ()
    return self:ContainsAnyOfJTJpSchedules({A.PRECOMPUTE_J, A.PRECOMPUTE_J_THEN_JTJ}) 
end
function A.ScheduledEnergy:RequiresMatMul()
    return self:ContainsAnyOfJTJpSchedules({A.PRECOMPUTE_J_THEN_JTJ}) 
end
function A.ScheduledEnergy:RequiresJtJMaterialize()
    local requires = self:ContainsAnyOfJTJpSchedules({A.PRECOMPUTE_JTJ})
    return requires
end
function A.ScheduledEnergy:RequiresSeparateJtAndJ()
    local requires = self:ContainsAnyOfJTJpSchedules(
        {A.APPLY_SEPARATELY})
    return requires
end
function A.ScheduledEnergy:RequiresApplyJtJp()
    local requires = self:ContainsAnyOfJTJpSchedules(
        {A.INLINE}) 
    return requires
end
function A.ScheduledEnergy:RequiresJtJMatVecMul()
    local requires = self:ContainsAnyOfJTJpSchedules(
        {A.PRECOMPUTE_J_THEN_JTJ, A.PRECOMPUTE_JTJ}) 
    return requires
end

function A.ResidualGroup:type(name)
    local imtype = A.ImageType(self:domain():IndexSpace(), thallo_float, #self.residuals)
    return A.ResidualGroupType(name,imtype)
end

function A.ScheduledEnergy:ResidualType()
    if not self._ResidualType then
        local groups = List()
        for i,rg in ipairs(self.residualgroups) do
            local name = "residual_"..i
            groups:insert(rg:type(name))
        end
        self._ResidualType = A.ResidualType(groups)
    end
    return self._ResidualType
end

-- Forward these functions through the scheduleenergy parameter of ProblemSpec
local scheduledenergyfunctions = {"JTJpSchedules","CanFuseJtJpReduction", "CanFusePCGInit", "ContainsAnyOfJTJpSchedules", "RequiresJ", "RequiresMatMul", "RequiresJtJMaterialize", "RequiresSeparateJtAndJ", "RequiresApplyJtJp", "RequiresJtJMatVecMul", "ResidualType"}

for _,fnname in ipairs(scheduledenergyfunctions) do
    A.ProblemSpec[fnname] = function (self)
        self:Stage "functions"
        return self.scheduledenergy[fnname](self.scheduledenergy)
    end
end

function A.ProblemSpec:Stencil(stencil) 
    self:Stage "inputs"
    self.maxStencil = math.max(stencil, self.maxStencil)
end

function A.ProblemSpec:newparameter(p)
    assert(A.ProblemParam:isclassof(p))
    self:registername(p.name)
    self.parameters:insert(p)
end

function A.ProblemSpec:ParameterType()
    self:Stage "functions"
    if not self.ProblemParameters then
        self.ProblemParameters = terralib.types.newstruct("ProblemParameters")
        self.ProblemParameters.entries:insert { "X" , self:UnknownType():terratype() }
        for i,p in ipairs(self.parameters) do
            local n,t = p.name,p:terratype()
            if not p.isunknown then 
                self.ProblemParameters.entries:insert { n, t } 
            end
        end
    end
    return self.ProblemParameters
end

function A.ProblemSpec:UnknownType()
    self:Stage "functions"
    if not self._UnknownType then
        local images = util.filter(self.parameters, function(p) return p.isunknown end)
        self._UnknownType = A.UnknownType(images)
    end
    return self._UnknownType
end

function A.ProblemSpec:DirectSolve()
    self:Stage "functions"
    return self._direct_solve
end

function A.IterationDomain:IndexSpace()
    local dims = self.domains:map(function(d) return d.dim end)
    return A.IndexSpace(dims)
end

function A.ResidualGroup:domain()
    return self.domainandschedule.domain.external
end

function A.ResidualGroup:jacobianEntriesPerElement()
    if not self._nnz_per_entry then
        local domain_multipliers = {}
        local rg_domain = self.domainandschedule.domain
        for _,domain in ipairs(rg_domain.full.domains) do
            if table.indexOf(rg_domain.external.domains,domain) > 0 then
                domain_multipliers[domain] = 1
            else
                domain_multipliers[domain] = domain.dim.size
            end
        end
        self._nnz_per_entry = 0
        for _,r in ipairs(self.residuals) do
            for _,u in ipairs(r.unknowns) do
                local num_unknowns = 1
                local unknown_domains = util.extract_domains_from_index(u.index)
                for _,d in ipairs(unknown_domains) do
                    num_unknowns = num_unknowns*domain_multipliers[d]
                end
                self._nnz_per_entry = self._nnz_per_entry + num_unknowns
            end
        end
    end
    return self._nnz_per_entry
end

function A.ProblemSpec:JTJType()
    self:Stage "functions"
    if not self._JTJType then
        local blocks = List()
        local unknowns = self:UnknownType().images
        for _,u0 in ipairs(unknowns) do
            for _,u1 in ipairs(unknowns) do
                blocks:insert(A.JTJBlockType(u0:outerproduct_name(u1), u0.imagetype:outerproduct(u1.imagetype)))
            end
        end
        self._JTJType = A.JTJType(blocks)
    end
    return self._JTJType
end

function A.ProblemSpec:FlatUnknownwiseFunction()
    self:Stage "functions"
    if not self._FlatUnknownwiseFunction then
        self._FlatUnknownwiseFunction = A.GenericFunction(self:UnknownType():cardinality())
    end
    return self._FlatUnknownwiseFunction
end

function A.ProblemSpec:JTJwiseFunction()
    self:Stage "functions"
    if not self._JTJwiseFunction then
        self._JTJwiseFunction = A.GenericFunction(self:UnknownType():cardinality()*self:UnknownType():cardinality())
    end
    return self._JTJwiseFunction
end

function A.GenericFunction:__tostring() return "GenericFunction("..self.numelements..")" end
function A.UnknownwiseFunction:__tostring()
    return "UnknownwiseFunction("..reduce(op.concat, "", map(tostring, self.domain.domains))..")"
end
function A.UnknownwiseFunction:indextype()
    return self.domain:IndexSpace():indextype()
end

function array_to_string(arr,separator)
    if not separator then
        separator = ","
    end
    local str = "{ "
    if #arr > 1 then
        for i=1,#arr-1 do
            str = str..tostring(arr[i])..separator
        end
    end
    if #arr > 0 then
        str = str..tostring(arr[#arr]).." "
    end
    return str.."}"
end

function A.ResidualGroupSchedule:__tostring()
    local sched = ""
    if self.jtj_materialize then
        if self.j_materialize then
            sched = "[[Jt][J]]p"
        else
            sched = "[JtJ]p"
        end
    elseif self.j_materialize then
        sched = "[Jt][[J]p]"
    elseif self.jp_materialize then
        sched = "Jt[Jp]"
    else
        sched = "JtJp"
    end
    sched = sched..", "..array_to_string(self.domain_order)
    sched = sched..", JtJp "..(self.compute_at_output and "U" or "R")
    sched = sched..", JtF "..(self.jtf_compute_at_output and "U" or "R")
    return sched
end



function A.FullSchedule:__tostring() 
    local result = "FullSchedule {\n"
    for i=1,#self.residualnames do
        result = result.."  "..self.residualnames[i].." = {"
        result = result.."  "..tostring(self.rgschedules[i]).."  }\n" 
    end
    result = result.."  Inline Expressions: "..array_to_string(self.exptoinline) 
    result = result.."\n"
    result = result.."  Inline Gradient Expressions "..array_to_string(self.exptoinlinegradient) 
    result = result.."\n"
    result = result.."}\n"
    return result
end


function A.Image:indextype()
    return self.type.ispace:indextype()
end

function A.ResidualwiseFunction:__tostring()
    return "ResidualwiseFunction("..reduce(op.concat, "", map(tostring, self.domain.domains))..")"
end
function A.ResidualwiseFunction:cardinality()
    return foldl(function(acc,next) return acc*next.dim.size end, 1, self.domain.domains)
end
function A.ResidualwiseFunction:indextype()
    return self.domain:IndexSpace():indextype()
end

function A.ResidualAndContractionwiseFunction:__tostring()
    return "ResidualAndContractionwiseFunction("..reduce(op.concat, "", map(tostring, self.domain.domains))..")"
end
function A.ResidualAndContractionwiseFunction:cardinality()
    return foldl(function(acc,next) return acc*next.dim.size end, 1, self.domain.domains)
end
function A.ResidualAndContractionwiseFunction:indextype()
    return self.domain:IndexSpace():indextype()
end

function A.IterationDomainwiseFunction:__tostring()
    return "IterationDomainwiseFunction("..reduce(op.concat, "", map(tostring, self.domain.domains))..")"
end
function A.IterationDomainwiseFunction:cardinality()
    return foldl(function(acc,next) return acc*next.dim.size end, 1, self.domain.domains)
end
function A.IterationDomainwiseFunction:indextype()
    return self.domain:IndexSpace():indextype()
end

function A.GenericFunction:cardinality()
    return self.numelements
end

function A.IndexDomain:__tostring()
    return self.name
end

function A.ProblemSpec:Functions(ft, functions)
    self:Stage "functions"
    for k,v in pairs(functions) do
        if k ~= "derivedfrom" then
            v:gettype() -- typecheck now
        end
    end
    assert(A.FunctionKind:isclassof(ft))
    if not functions.exclude then
        functions.exclude = macro(function() return `false end)
    end
    self.functions:insert(A.ProblemFunctions(ft, functions))
end

function A.ProblemSpec:Param(name,typ,idx)
    self:Stage "inputs"
    self:newparameter(A.ScalarParam(typ,name,idx))
end

function A.ProblemSpec:UsesLambda() return self.problemkind:match("LM") ~= nil end

function A.Dim:__tostring() return "Dim("..self.name..")" end

function A.Dim:__call()
    local domainindex = self.domaincount
    self.domaincount = domainindex + 1
    local name = self.name
    if domainindex > 1 then
        name = name.."_"..tostring(domainindex)
    end
    local domain = A.IndexDomain(self, name, index_domain_index) 
    index_domain_index = index_domain_index + 1
    return domain
end

function tocomponent(a)
    if A.IndexDomain:isclassof(a) then
        return A.DirectIndexComponent(a)
    elseif A.SparseAccess:isclassof(a) then
        return A.SparseIndexComponent(a)
    elseif type(a) == "number" then
        return A.ConstantIndexComponent(a)
    else
        assert(A.IndexComponent:isclassof(a), "Invalid Index Component: "..tostring(a))
        return a
    end
end

function isoffsetcomponent(comp)
    if A.DirectIndexComponent:isclassof(comp) then
        return true
    end
    if A.ConstantIndexComponent:isclassof(comp.rhs) then
        if A.DirectIndexComponent:isclassof(comp.lhs) then
            return true
        end
    end
    return false
end

function isnormalizedcomponent(comp)
    return A.DirectIndexComponent:isclassof(comp) or (isoffsetcomponent(comp) and comp.op == "+")
end

function normalizedcomponenttodomainandshift(comp)
    assert(isnormalizedcomponent(comp))
    if A.DirectIndexComponent:isclassof(comp) then
        return comp.domain,0
    else 
        return comp.lhs.domain,comp.rhs.value
    end
end

function A.ConstantIndexComponent:__unm()
    return A.ConstantIndexComponent(-self.value)
end

-- TODO: rewrite this to be more robust
function normalizeindexcomponent(comp)
    assert(A.ComponentBinOp:isclassof(comp))
    if comp.op == "-" then
        if A.ConstantIndexComponent:isclassof(comp.rhs) then
            comp = A.ComponentBinOp("+", comp.lhs, -comp.rhs)
	    end
	end
    if comp.op == "+" and A.ConstantIndexComponent:isclassof(comp.lhs) and A.DirectIndexComponent:isclassof(comp.rhs) then
        comp = A.ComponentBinOp("+", comp.rhs, comp.lhs)
    end
    if A.ComponentBinOp:isclassof(comp.lhs) and comp.lhs.op == "+" and A.ConstantIndexComponent:isclassof(comp.lhs.rhs) then
        comp = A.ComponentBinOp(comp.op, comp.lhs.lhs, A.ConstantIndexComponent(comp.lhs.rhs.value + comp.rhs.value))
    end
    if comp.rhs.value == 0 then
        comp = comp.lhs
    end
    return comp
end

function addindexcomponents(a,b)
    a = tocomponent(a)
    b = tocomponent(b)
    return normalizeindexcomponent(A.ComponentBinOp("+", a, b))
end
function subindexcomponents(a,b)
    a = tocomponent(a)
    b = tocomponent(b)
    return normalizeindexcomponent(A.ComponentBinOp("-", a, b))
end

function A.IndexComponent.__add(a,b)
    return addindexcomponents(a,b)
end

function A.IndexComponent.__sub(a,b)
    return subindexcomponents(a,b)
end

function A.IndexDomain.__add(a,b)
    return addindexcomponents(a,b)
end

function A.IndexDomain.__sub(a,b)
    return subindexcomponents(a,b)
end
-- TODO: assert sparse access only has 1 component
function A.SparseAccess.__add(a,b)
    return addindexcomponents(a,b)
end
function A.SparseAccess.__sub(a,b)
    return subindexcomponents(a,b)
end

thallo.dimcount = 0

function thallo.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(thallo.dimensions[idx])
    if thallo.dimcount < idx+1 then thallo.dimcount = idx+1 end
    local dim = A.Dim(name,size,idx)
    dim.domaincount = 0
    return dim
end

function A.IndexSpace:DimCount()
    return #self.dims
end

function A.IndexSpace:cardinality()
    local c = 1
    for i,d in ipairs(self.dims) do
        c = c * d.size
    end
    return c
end
function A.IndexSpace:init()
    self._string = self.dims:map(function(x) return x.name end):concat("_")
end
function A.IndexSpace:__tostring() return self._string end

function A.IndexSpace:getiterationdomain()
    if not self._iterationdomain then
        self._iterationdomain = A.IterationDomain(self.dims:map(function(d) return d() end))
    end
    return self._iterationdomain
end

function A.IndexSpace:indextype()
    if self._terratype then return self._terratype end
    local dims = self.dims
    assert(#dims > 0, "index space must have at least 1 dimension")
    local struct Index {}
    self._terratype = Index
    
    local params,params2 = List(),List()
    local fieldnames = List()
    for i = 1,#dims do
        local n = "d"..tostring(i-1)
        params:insert(symbol(int,n))
        params2:insert(symbol(int,n))
        fieldnames:insert(n)
        Index.entries:insert { n, int }
    end

    terra Index.metamethods.__apply(self : &Index, [params])
        var rhs : Index
        escape
            for i = 1,#dims do
                emit quote  
                    rhs.[fieldnames[i]] = self.[fieldnames[i]] + [params[i]]
                end 
            end
        end
        return rhs
    end

    terra Index.metamethods.__eq(self : &Index, other : Index)
        var equal = true
        escape
            for i = 1,#dims do
                emit quote  
                    equal = equal and (self.[fieldnames[i]] == other.[fieldnames[i]])
                end 
            end
        end
        return equal
    end


    terra Index:wrap()
        escape
            for i = 1,#dims do
                emit quote
                    if self.[fieldnames[i]] < 0 then
                        self.[fieldnames[i]] = self.[fieldnames[i]] + [dims[i].size]
                    elseif self.[fieldnames[i]] >= [dims[i].size] then
                        self.[fieldnames[i]] = self.[fieldnames[i]] - [dims[i].size]
                    end
                end
            end
        end
        return self
    end

    local function genoffset(self)
        local s = 1
        local offset = `self.d0
        for i = 2,#dims do
            s = s * dims[i-1].size
            offset = `[C.size_t](s)*self.[fieldnames[i]] + offset
        end
        return offset
    end
    terra Index:tooffset()
        return [genoffset(self)]
    end
    local function genbounds(self,bmins,bmaxs)
        local valid
        for i = 1, #dims do
            local n = fieldnames[i]
            local bmin,bmax = 0,0
            if bmins then
                bmin = assert(bmins[i])
            end
            if bmaxs then
                bmax = assert(bmaxs[i])
            end
            local v = `self.[n] >= -[bmin] and self.[n] < [dims[i].size] - [bmax]
            if valid then
                valid = `valid and v
            else
                valid = v
            end
        end
        return valid
    end
    terra Index:InBounds() return [ genbounds(self) ] end
    terra Index:InBoundsExpanded([params],[params2]) return [ genbounds(self,params,params2) ] end
    if #dims <= 3 then
        local dimnames = "xyz"
        terra Index:initFromCUDAParams() : bool
            escape
                local lhs,rhs = terralib.newlist(),terralib.newlist()
                local valid = `true
                for i = 1,#dims do
                    local name = dimnames:sub(i,i)
                    local l = `self.[fieldnames[i]]
                    local r = `blockDim.[name] * blockIdx.[name] + threadIdx.[name]
                    lhs:insert(l)
                    rhs:insert(r)
                    valid = `valid and l < [dims[i].size]
                end
                emit quote
                    [lhs] = [rhs]
                    return valid
                end
            end  
        end
    end

    Index.maxlinearindex = self:cardinality()
    -- TODO: optimize
    terra Index:init(linearIdx : int32)
        escape
            local divisor = 1
            for i = 1,#dims do
                emit quote self.[fieldnames[i]] = (linearIdx / [divisor]) % [dims[i].size] end
                divisor = divisor * dims[i].size
            end
        end
        return linearIdx < [Index.maxlinearindex]
    end
    Index.ispace = self
    Index.infostring = "from: IndexSpace("..tostring(self)..")"

    return Index
end

function A.ImageType:usestexture() -- texture, 2D texture
    return false, false 
end

local wrapBindlessTexture = nil

local cd = util.cd


function A.ImageType:cardinality()
    return self.ispace:cardinality() * self.channelcount
end

function A.ImageType:indextype()
    return self.ispace:indextype()
end

function A.ImageType:ElementType() return util.Vector(self.scalartype,self.channelcount) end
function A.ImageType:LoadAsVector() return self.channelcount == 2 or self.channelcount == 4 end
function A.ImageType:terratype()
    if self._terratype then return self._terratype end
    local scalartype = self.scalartype
    local vectortype = self:ElementType()
    local struct Image {
        data : &vectortype
    }
    self._terratype = Image
    local channelcount = self.channelcount

    local Index = self.ispace:indextype()
    self.indextype = Index
    local imispace = self.ispace
    function Image.metamethods.__typename()
      return string.format("Image(%s,%s,%d)",tostring(self.scalartype),tostring(self.ispace),channelcount)
    end

    local cardinality = self.ispace:cardinality()
    local VT = &vector(scalartype,channelcount)
    
    terra Image:totalbytes() return sizeof(vectortype)*cardinality end
    -- reads
    if stub_out_image_loads then
        terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
            if [print_image_indices] then printf([Image.metamethods.__typename().." Dummy load from: %d\n"], idx:tooffset()) end
            var v : vectortype = 0
            return v
        end
    else
        if self:LoadAsVector() then
            terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
                var a = VT(self.data)[idx:tooffset()]
                return @[&vectortype](&a)
            end
        else
            terra Image.metamethods.__apply(self : &Image, idx : Index) : vectortype
                return self.data[idx:tooffset()]
            end
        end
    end

    terra Image:rawScalarPtr()
        return [&scalartype](self.data)
    end

    terra Image:rawScalarSet(idx : int, v : scalartype)
        [&scalartype](self.data)[idx] = v
    end

    terra Image:rawScalarGet(idx : int) : scalartype
        return [&scalartype](self.data)[idx]
    end

    local scalarCount = cardinality*channelcount
    terra Image:printDump()
        C.printf([Image.metamethods.__typename().."\n"])
        var hdata = [&thallo_float](C.malloc(self:totalbytes()))
        cd(C.cudaMemcpy(hdata, self:rawScalarPtr(), self:totalbytes(), C.cudaMemcpyDeviceToHost))
        for i = 0, [uint64](scalarCount) do
            C.printf("\t%d: %g\n",i,hdata[i])
        end
        C.free(hdata)
    end

    -- writes
    if self:LoadAsVector() then
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            VT(self.data)[idx:tooffset()] = @VT(&v)
        end
    else
        terra Image.metamethods.__update(self : &Image, idx : Index, v : vectortype)
            self.data[idx:tooffset()] = v
        end
    end

    if scalartype == thallo_float then    
        if deatomize then
            terra Image:atomicAddChannel(idx : Index, c : int32, v : scalartype)
                var addr : &scalartype = &self.data[idx:tooffset()].data[c]
                addr[0] = v
            end
            terra Image:aggregatedAtomicAddChannel(idx : Index, c : int32, v : scalartype, peers : uint)
                self:atomicAddChannel(idx,c,v)
            end
        else
            terra Image:aggregatedAtomicAddChannel(idx : Index, c : int32, v : scalartype, peers : uint)
                var key : int = idx:tooffset()
                var addr : &scalartype = &self.data[key].data[c]
                util.warp_aggregated_atomic_reduction_by_key(addr, peers, v)
            end
            terra Image:atomicAddChannel(idx : Index, c : int32, v : scalartype)
                var addr : &scalartype = &self.data[idx:tooffset()].data[c]
                util.atomicAdd(addr,v)
            end
        end
        terra Image:atomicAdd(idx : Index, v : vectortype) -- only for hand written stuff
            for i = 0,channelcount do
                self:atomicAddChannel(idx,i,v(i))
            end
        end
        terra Image:setChannel(idx : Index, c : int32, v : scalartype)
            if [guard_atomics] and not idx:InBounds() then
                return
            end
            self.data[idx:tooffset()].data[c] = v
        end
    end

    if stub_out_image_loads then
        terra Image:get(idx : Index)
            if idx:InBounds() then
                if [print_image_indices] then printf([Image.metamethods.__typename().." Dummy get from: %d\n"], idx:tooffset()) end
            end
            var v : vectortype = 0.f
            return v
        end
    else
        terra Image:get(idx : Index)
            var v : vectortype = 0.f
            if idx:InBounds() then
                v = self(idx)
            end
            return v
        end
    end
    local terra lerp(v0 : vectortype, v1 : vectortype, t : thallo_float)
        return (thallo_float(1.) - t)*v0 + t*v1
    end

    -- lerps for 2D images only
    if 2 == #self.ispace.dims then
        if stub_out_image_loads then
            terra Image:sample(x : thallo_float, y : thallo_float)
                if [print_image_indices] then
                    printf([Image.metamethods.__typename().." Dummy load from: %f, %f\n"], x,y)
                end
                var v : vectortype = 0.0
                return v
            end
        else
            terra Image:sample(x : thallo_float, y : thallo_float)
                if [print_image_indices] then printf([Image.metamethods.__typename().." sample from: %f, %f\n"], x,y) end
                var x0 : int, x1 : int = thallo.math.floor(x),thallo.math.ceil(x)
                var y0 : int, y1 : int = thallo.math.floor(y),thallo.math.ceil(y)
                var xn,yn = x - x0,y - y0
                var u = lerp(self:get( Index {x0,y0} ),self:get( Index {x1,y0} ),xn)
                var b = lerp(self:get( Index {x0,y1} ),self:get( Index {x1,y1} ),xn)
                return lerp(u,b,yn)
            end
        end
    end
    if 3 == #self.ispace.dims then
        if stub_out_image_loads then
            terra Image:sample(x : thallo_float, y : thallo_float, z : int)
                if [print_image_indices] then
                    printf([Image.metamethods.__typename().." Dummy load from: %f, %f, %d\n"], x,y,z)
                end
                var v : vectortype = 0.0
                return v
            end
        else
            if false then
                terra Image:sample(x : thallo_float, y : thallo_float, z : int)
                    if [print_image_indices] then printf([Image.metamethods.__typename().." sample from: %f, %f, %d\n"], x,y,z) end
                    var x0 : int, x1 : int = thallo.math.floor(x),thallo.math.ceil(x)
                    var y0 : int, y1 : int = thallo.math.floor(y),thallo.math.ceil(y)
                    var xn,yn = x - x0,y - y0
                    var u = lerp(self:get( Index {x0,y0,z} ),self:get( Index {x1,y0,z} ),xn)
                    var b = lerp(self:get( Index {x0,y1,z} ),self:get( Index {x1,y1,z} ),xn)
                    return lerp(u,b,yn)
                end
            else
                terra Image:horizontalConditionalLerp(s : &vectortype, w : &float, x : int, y : int, z : int, alpha : float, imageWidth : int, imageHeight : int)
                    if (x >= 0 and y >= 0 and x < imageWidth and y < imageHeight) then
                        var v : vectortype = self(Index {x,y,z})
                        if (v(0) ~= [-math.huge]) then 
                            @s = @s + (alpha*v); 
                            @w = @w + alpha; 
                        end 
                    end
                end
                -- TODO(mmara): Factor out texture sampling from the rest of the code
                terra Image:sample(x : thallo_float, y : thallo_float, z : int)
                    if [print_image_indices] then printf([Image.metamethods.__typename().." sample from: %f, %f, %d\n"], x,y,z) end
                    
                    var imageWidth : int = [imispace.dims[1].size]
                    var imageHeight : int = [imispace.dims[2].size]
                    var x0 : int, x1 : int = thallo.math.floor(x),thallo.math.ceil(x)
                    var y0 : int, y1 : int = thallo.math.floor(y),thallo.math.ceil(y)
                    var alpha = x - x0
                    var beta = y - y0

                    var s0 : vectortype = 0.0f
                    var w0 : float = 0.0f
                    self:horizontalConditionalLerp(&s0, &w0, x0, y0, z, (1.0f - alpha), imageWidth, imageHeight)
                    self:horizontalConditionalLerp(&s0, &w0, x1, y0, z, alpha, imageWidth, imageHeight)
                    
                    var s1 : vectortype = 0.0f
                    var w1 : float = 0.0f
                    self:horizontalConditionalLerp(&s1, &w1, x0, y1, z, (1.0f - alpha), imageWidth, imageHeight)
                    self:horizontalConditionalLerp(&s1, &w1, x1, y1, z, alpha, imageWidth, imageHeight)

                    var p0 = s0 * (1.0f/w0)
                    var p1 = s1 * (1.0f/w1)

                    var ss : vectortype = 0.0f 
                    var ww : float = 0.0f
                    if (w0 > 0.0f) then
                        ss = ss + (1.0f - beta)*p0
                        ww = ww + (1.0f - beta)
                    end
                    if (w1 > 0.0f) then
                        ss = ss + beta * p1
                        ww = ww + beta
                    end

                    if (ww > 0.0f) then
                        return ss / ww
                    else
                        return vectortype([-math.huge])
                    end
                end
                terra Image:access(x : int, y : int, z : int)
                    self:get( Index {x,y,z} )
                end

            end
        end
    end

    terra Image:clear()
        logTrace([Image.metamethods.__typename()..":clear() (%p) (%lu)\n"], [&opaque](self.data), self:totalbytes())
        cd(C.cudaMemsetAsync([&opaque](self.data), 0, self:totalbytes(), nil))
    end

    
    terra Image:setGPUptr(ptr : &uint8) self.data = [&vectortype](ptr) end
    terra Image:freeData()
        logTrace([Image.metamethods.__typename()..":freeData()\n"])
        if self.data ~= nil then
            cd(C.cudaFree([&opaque](self.data)))
            self.data = nil
        end
    end

    terra Image:initFromGPUptr( ptr : &uint8 )
        logTrace([Image.metamethods.__typename()..":initFromGPUptr(%p)\n"],ptr)
        self.data = nil
        self:setGPUptr(ptr)
    end
    terra Image:initGPU()
        logTrace([Image.metamethods.__typename()..":initGPU()\n"])
        var data : &uint8
        cd(C.cudaMalloc([&&opaque](&data), self:totalbytes()))
        self:initFromGPUptr(data)
        self:clear()
    end
    return Image
end

local function MapAndGroupBy(list,fn,...)
    local groups,map = List(),{}
    for _,l in ipairs(list) do
        local g,v = fn(l,...)
        if not map[g] then
            map[g] = List()
            groups:insert(g)
        end
        map[g]:insert(v)
    end
    return groups,map
end

local function MapAndGroupByI(list,fn,...)
    local groups,map = List(),{}
    for i,l in ipairs(list) do
        local g,v = fn(i,l,...)
        if not map[g] then
            map[g] = List()
            groups:insert(g)
        end
        map[g]:insert(v)
    end
    return groups,map
end

function A.UnknownType:init()
    self.ispaces,self.ispacetoimages = MapAndGroupBy(self.images, function(ip)
        if thallo_float == float then
            assert(ip.imagetype.scalartype == float, "Unknowns must be floats when doublePrecision = false, but "..ip.name.." was declared as "..tostring(ip.imagetype.scalartype))
        else
            assert(thallo_float == double, "Invalid unknown type "..tostring(thallo_float))
            assert(ip.imagetype.scalartype == double, "Unknowns must be doubles when doublePrecision = true, but "..ip.name.." was declared as "..tostring(ip.imagetype.scalartype))
        end
        return ip.imagetype.ispace, ip
    end)
    self.ispacesizes = {}
    for _,ispace in ipairs(self.ispaces) do
        local N = 0
        for _,ip in ipairs(self.ispacetoimages[ispace]) do
            N = N + ip.imagetype.channelcount
        end
        self.ispacesizes[ispace] = N
    end
end
function A.UnknownType:IndexSpaces()
    return self.ispaces
end
function A.UnknownType:VectorSizeForIndexSpace(ispace) return assert(self.ispacesizes[ispace],"unused ispace: "..tostring(ispace)) end 
function A.UnknownType:VectorTypeForIndexSpace(ispace)
    return util.Vector(thallo_float,self:VectorSizeForIndexSpace(ispace))
end

function A.UnknownType:UnknownIteratorForIndexSpace(ispace)
    local images = self.ispacetoimages[ispace]
    local i,j,c = 0,1,0
    return function()
        if c >= images[j].imagetype.channelcount then
            j,c = j+1,0
        end
        if j > #images then return nil end
        i,c = i + 1,c + 1
        return i - 1, images[j].name,c - 1
    end
end

function A.UnknownType:cardinality()
    local cardinality = 0
    local unknowns = self.images
    for _,u in ipairs(unknowns) do
        cardinality = cardinality + u.imagetype:cardinality()
    end
    return cardinality
end

function A.UnknownType:terratype()
    if self._terratype then return self._terratype end
    self._terratype = terralib.types.newstruct("UnknownType")
    local T = self._terratype
    local images = self.images
    for i,ip in ipairs(images) do
        T.entries:insert { ip.name, ip.imagetype:terratype() }
    end
    if use_contiguous_allocation then
        T.entries:insert { "_contiguousallocation", &opaque }
        terra T:initGPU()
            var size = 0
            escape
                for i,ip in ipairs(images) do
                    emit quote 
                        size = size + self.[ip.name]:totalbytes()
                    end
                end
            end
            var data : &uint8
            cd(C.cudaMalloc([&&opaque](&data), size))
            self._contiguousallocation = data
            cd(C.cudaMemsetAsync([&opaque](data), 0, size, nil))
            size = 0
            escape
                for i,ip in ipairs(images) do
                    emit quote 
                        self.[ip.name]:initFromGPUptr(data+size)
                        size = size + self.[ip.name]:totalbytes() 
                    end
                end
            end
        end
        terra T:freeData()
            cd(C.cudaFree(self._contiguousallocation))
        end
    else
        terra T:initGPU()
            escape
                for i,ip in ipairs(images) do
                    emit quote logTrace(["UnknownType:"..ip.name..":initGPU()\n"]) end
                    emit quote self.[ip.name]:initGPU() end
                end
            end
        end
        terra T:freeData()
            escape
                for i,ip in ipairs(images) do
                    emit quote self.[ip.name]:freeData() end
                end
            end
        end
    end
    terra T:clear()
        escape
            for i,ip in ipairs(images) do
                emit quote logTrace(["UnknownType:"..ip.name..":clear()\n"]) end
                emit quote self.[ip.name]:clear() end
            end
        end
    end

    -- Using macros evaluated at typechecking to implement polymorphism on the index type
    -- See pfn() in section 7.4 of "First-class Runtime Generation of High-performance 
    -- Types using Exotypes", http://terralang.org/pldi083-devito.pdf

    local apply = {}
    local atomicAdd = {}
    local update = {}

    for _,ispace in ipairs(self:IndexSpaces()) do   
        local Index = ispace:indextype()
        local ispaceimages = self.ispacetoimages[ispace]
        local VT = self:VectorTypeForIndexSpace(ispace)
        apply[Index] = terra (self : &T, idx : Index) : VT
            var r : VT
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d = self.[im.name](idx)
                        for i = 0,im.imagetype.channelcount do
                            r.data[off+i] = d.data[i]
                        end
                    end
                    off = off + im.imagetype.channelcount
                end
            end
            return r
        end
        atomicAdd[Index] = terra (self : &T, idx : Index, val : VT)
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        for i = 0,im.imagetype.channelcount do
                            self.[im.name]:atomicAddChannel(idx, i, val.data[off+i])
                        end
                    end
                    off = off + im.imagetype.channelcount
                end
            end
        end
        update[Index] = terra (self : &T, idx : Index, v : VT)
            escape
                local off = 0
                for _,im in ipairs(ispaceimages) do
                    emit quote
                        var d : im.imagetype:ElementType()
                        for i = 0,im.imagetype.channelcount do
                            d.data[i] = v.data[off+i]
                        end
                        self.[im.name](idx) = d
                    end
                    off = off + im.imagetype.channelcount
                end
            end
        end
    end
    T.metamethods.__apply = macro(function(self, idx)
        return `[apply[idx:gettype()]](&self, idx)
    end)

    T.methods.atomicAdd = macro(function(self, idx, val)
        return `[atomicAdd[idx:gettype()]](&self, idx, val)
    end)

    T.metamethods.__update = macro(function(self, idx, v)
        return `[update[idx:gettype()]](&self, idx, v)
    end)


    return self._terratype
end



function A.ResidualType:terratype()
    if self._terratype then return self._terratype end
    self._terratype = terralib.types.newstruct("ResidualType")
    local T = self._terratype
    local rgroups = self.groups
    for i,rg in ipairs(rgroups) do
        T.entries:insert { rg.name, rg.imagetype:terratype() }
    end

    terra T:printDump()
        escape
            for _,rg in ipairs(rgroups) do
                emit quote C.printf(["ResidualType: "..rg.name.."{\n"]) end
                emit quote self.[rg.name]:printDump() end
            end
        end
    end

    terra T:initGPU()
        escape
            for _,rg in ipairs(rgroups) do
                emit quote logTrace(["ResidualType:"..rg.name..":initGPU()\n"]) end
                emit quote self.[rg.name]:initGPU() end
            end
        end
    end
    terra T:freeData()
        escape
            for _,rg in ipairs(rgroups) do
                emit quote self.[rg.name]:freeData() end
            end
        end
    end
    terra T:clear()
        escape
            for _,rg in ipairs(rgroups) do
                emit quote logTrace(["ResidualType:"..rg.name..":clear()\n"]) end
                emit quote self.[rg.name]:clear() end
            end
        end
    end
    return self._terratype
end

local function tovalidimagetype(typ)
    if not terralib.types.istype(typ) then return nil end
    if util.isvectortype(typ) then
        return typ.metamethods.type, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end


function A.JTJType:terratype()
    if self._terratype then return self._terratype end
    self._terratype = terralib.types.newstruct("JTJType")
    local T = self._terratype
    local groups = self.blocks

    for _,g in ipairs(groups) do
        T.entries:insert { g.name, g.imagetype:terratype() }
    end
    
    terra T:initGPU()
        escape
            for _,g in ipairs(groups) do
                emit quote logTrace(["JTJType:"..g.name..":initGPU()\n"]) end
                emit quote self.[g.name]:initGPU() end
            end
        end
    end
    terra T:freeData()
        escape
            for _,g in ipairs(groups) do
                emit quote self.[g.name]:freeData() end
            end
        end
    end
    terra T:clear()
        escape
            for _,g in ipairs(groups) do
                emit quote logTrace(["JTJType:"..g.name..":clear()\n"]) end
                emit quote self.[g.name]:clear() end
            end
        end
    end

    return self._terratype
end

function A.ProblemSpec:ImageType(typ,ispace)
    local scalartype,channelcount = tovalidimagetype(typ,"expected a number or an array of numbers")
    assert(scalartype,"expected a number or an array of numbers")
    return A.ImageType(ispace,scalartype,channelcount) 
end

local function toispace(ispace)
    if not A.IndexSpace:isclassof(ispace) then -- for handwritten API
        assert(#ispace > 0, "expected at least one dimension")
        ispace = A.IndexSpace(List(ispace)) 
    end
    return ispace
end


function A.ProblemSpec:Image(name,typ,ispace,idx,isunknown)
    self:Stage "inputs"
    isunknown = isunknown and true or false
    self:newparameter(A.ImageParam(self:ImageType(typ,toispace(ispace)),isunknown,name,idx))
end
function A.ProblemSpec:Unknown(name,typ,ispace,idx) return self:Image(name,typ,ispace,idx,true) end


function A.ProblemSpec:Sparse(name, inspace, outspace, idx)
    self:Stage "inputs"
    local SparseType = terralib.types.newstruct(name)
    local mm = SparseType.metamethods
    mm.elements = terralib.newlist()
    local ospace = toispace(outspace)
    assert(#ospace.dims == 1)
    self:newparameter(A.SparseParam(toispace(inspace),toispace(outspace),name,assert(tonumber(idx))))
end

local activePlans = terralib.newlist()

errorPrint = rawget(_G,"errorPrint") or print

function thallo.problemSpecFromFile(filename,exauto_index,lin_iter_hint)
    local file, errorString = terralib.loadfile(filename)
    if not file then
        error(errorString, 0)
    end
    local P = ad.ProblemSpec()
    local libinstance = thallolib(P)
    setfenv(file,libinstance)
    local result = file()
    if not A.ProblemSpec:isclassof(result) then
        result = libinstance.Result(exauto_index, lin_iter_hint)
    end
    assert(result == nil or A.ProblemSpec:isclassof(result), "Loaded terra file was not a problem spec!")
    return result
end

local function printCurrentBytes()
    collectgarbage()
    collectgarbage()
    print(collectgarbage("count"))
end

thallo.compilePlan = compilePlan
thallo.math = util.gpuMath

local function problemPlan(id, dimensions, pplan, exauto_index, lin_iter_hint)
    local success,p = xpcall(function()  
        local profiler
        if profile_plan then 
            profiler = newProfiler("sampling")
            profiler:start()
        end
        local problemmetadata = assert(problems[id])
        thallo.dimensions = dimensions
        thallo.math = problemmetadata.kind:match("GPU") and util.gpuMath or util.cpuMath
        thallo.problemkind = problemmetadata.kind
        local b = terralib.currenttimeinseconds()
        if exauto_index >= 0 then
            ComputedArrayCache = {}
            computed_array_count = 0
            index_domain_index = 0
        end
        local tbl = thallo.problemSpecFromFile(problemmetadata.filename, exauto_index, lin_iter_hint)
        if not tbl then
            return 0
        end
        thallo.dimensions = {}
        for i=0,thallo.dimcount do
            thallo.dimensions[i] = dimensions[i]
        end
        assert(A.ProblemSpec:isclassof(tbl))
        local result = compilePlan(tbl,problemmetadata.kind,dimensions)
        local e = terralib.currenttimeinseconds()
        print("compile time: ", e - b)
        for i=0,thallo.dimcount do
            dimensions[i] = thallo.dimensions[i]
        end
        pplan[0] = result()
        activePlans[tostring(pplan[0])] = result
        pplan[0][0].problemid = id
        pplan[0][0].dimensions = dimensions
        for i=0,thallo.dimcount do
            dimensions[i] = thallo.dimensions[i]
        end
        -- TODO(mmara): Track down dimension memory corruption during cuda compilation and makePlan()
        if profile_plan then 
            profiler:stop()
            local outfile = io.open( "profile.txt", "w+" )
            profiler:report( outfile, true )
            outfile:close()
        end
        return 1
    end,function(err) errorPrint(debug.traceback(err,2)) end)
    return p
end
problemPlan = terralib.cast({int,&uint32,&&thallo.Plan,int,int} -> int, problemPlan)

local function planFree(pplan)
    local success,p = xpcall(function()
        activePlans[tostring(pplan)] = nil
        collectgarbage()
        collectgarbage()
    end,function(err) errorPrint(debug.traceback(err,2)) end)
end
planFree = terralib.cast({&thallo.Plan} -> {}, planFree)

function A.ImageIndex:__tostring() 
    return string.format("%s",self.components:map(function(x) return tostring(x) end):concat(",")) 
end

function A.DirectIndexComponent:Invert()
    return self
end

function A.ComponentBinOp:Invert()
    assert(isnormalizedcomponent(self))
    return A.ComponentBinOp(self.op,self.lhs,-self.rhs)
end

function A.ImageIndex:Invert()
    local components = List()
    for _,c in ipairs(self.components) do
        assert(isnormalizedcomponent(c))
        components:insert(c:Invert())
    end
    return A.ImageIndex(components)
end

function A.DirectIndexComponent:__tostring() return self.domain.name end
function A.SparseIndexComponent:__tostring() return tostring(self.access) end
function A.ConstantIndexComponent:__tostring() return tostring(self.value) end
function A.ComponentBinOp:__tostring() return ("(%s %s %s)"):format(self.lhs, self.op, self.rhs) end

function A.SparseAccess:__tostring() return ("%s(%s)"):format(tostring(self.sparse), self.index) end

function A.VarDef:asvar() return ad.v[self] end

function A.ImageAccess:__tostring()
    local r = ("%s(%s)(%s)"):format(self.image.name,tostring(self.index),self.channel)
    return r
end
function A.BoundsAccess:__tostring() return ("in[%s,%s]"):format(tostring(self.min),self.min == self.max and "p" or tostring(self.max)) end
function A.IndexValue:__tostring()
    local offsetstring = (self.shift_ == 0) and "" or "+"..tostring(self.shift_)
    return ("%s_%s"):format(tostring(self.indexdomain),offsetstring)  
end
function A.SparseIndexValue:__tostring() 
    local offsetstring = (self.shift_ == 0) and "" or "+"..tostring(self.shift_)
    return ("%s_%s%s"):format(tostring(self.access),tostring(self.index),offsetstring) 
end
function A.ParamValue:__tostring() return "param_"..self.name end


function A.DirectIndexComponent:substitute(src, dst)
    for i,domain in ipairs(src) do
        if domain == self.domain then
            return dst[i]
        end
    end
    return self
end

function A.ConstantIndexComponent:substitute(src, dst)
    return self
end

function A.SparseIndexComponent:substitute(src, dst)
    local internal_index = self.access.index:substitute(src,dst)
    return A.SparseIndexComponent(A.SparseAccess(self.access.sparse, internal_index))
end

function A.ComponentBinOp:substitute(src, dst)
    return normalizeindexcomponent(A.ComponentBinOp(self.op,self.lhs:substitute(src, dst),self.rhs:substitute(src, dst)))
end

function A.ImageIndex:substitute(src, dst)
    local newcomponents = List()
    for i,c in ipairs(self.components) do
        newcomponents:insert(c:substitute(src,dst))
    end
    return A.ImageIndex(newcomponents)
end

function A.ImageAccess:substitute(src, dst)
    return A.ImageAccess(self.image, self.index:substitute(src,dst), self.channel)
end

function A.IndexValue:substitute(src, dst)
    local newindexdomain = self.indexdomain
    local newshift = self.shift_
    for i,domain in ipairs(src) do
        if domain == self.indexdomain then
            local domain,shift = normalizedcomponenttodomainandshift(dst[i])
            return A.IndexValue(domain,shift+self.shift_)
        end
    end
    return self
end

local emptygradient = {}
function A.ImageAccess:gradient()
    local grad = {}
    if self.image.tensor_contraction then
        local exp,unks = self.image.expression, self.image.unknowns
        --print("Tensor contraction!!")
        --print(exp)
        local derivs = exp:gradient(unks:map(function(x) return ad.v[x] end))
        for i,d in ipairs(derivs) do
            grad[unks[i]] = d
        end
        return grad
    end
    local gradimage = self.image.gradientimage
    if gradimage then
        for i,unknown in ipairs(gradimage.unknowns) do
            local k = unknown:substitute(self.image.indexdomains,self.index.components)
            local expression = gradimage.image.expressions[i]
            --print("Gradient index "..tostring(self.index))
            local v = (ad.Const:isclassof(expression)) and expression or gradimage.image(self.index)(i-1)
            grad[k] = v
        end
        return grad
    end
    return emptygradient
end

function A.Image:Materialized(index)
    if self.inline then return false end
    local exp = self.expression
    if index then
        assert(self.expressions, "Error, checking if image without expressions was materialized: "..tostring(self))
        exp = self.expressions[index]
    end
    assert(exp, "invalid expression")
    return not ad.Const:isclassof(exp)
end

function A.IndexDomain:asvalue() return A.IndexValue(self,0):asvar() end
--TODO(mmara): Allow getting multidimensional graph accesses
function A.SparseAccess:asvalue() return A.SparseIndexValue(self,0,0):asvar() end

function ad.ProblemSpec()
    local ps = A.ProblemSpecAD()
    active_problem_spec = ps
    ps.P,ps.nametoimage,ps.precomputed,ps.extraunknownarguments,ps.extraresidualarguments,ps.excludeexps = thallo.ProblemSpec(), {}, List(), List(), List(), List()
    if ps.P:UsesLambda() then
        ps.trust_region_radius = ps:Param("trust_region_radius",thallo_float,-1)
        ps.radius_decrease_factor = ps:Param("radius_decrease_factor",thallo_float,-1)
        ps.min_lm_diagonal = ps:Param("min_lm_diagonal",thallo_float,-1)
        ps.max_lm_diagonal = ps:Param("max_lm_diagonal",thallo_float,-1)
    end
    return ps
end

function A.ProblemSpecAD:SetScheduledEnergy(scheduledenergy)
    self.P.scheduledenergy = scheduledenergy
    self.extraresidualarguments = List()
    self.extraunknownarguments = List()
end

function A.ProblemSpecAD:UsesLambda() return self.P:UsesLambda() end
function A.ProblemSpecAD:UsePreconditioner(v)
    self.P:UsePreconditioner(v)
end

function A.Image:init()
    if self.location == A.UnknownLocation then
        self.excludeexps = List()
    end
end

function A.ProblemSpecAD:Array(name,typ,dims,idx,isunknown)
    if not terralib.types.istype(typ) then
        typ,dims,idx,isunknown = thallo_float,typ,dims,idx --shift arguments left
    end
    isunknown = isunknown and true or false
    local ispace = toispace(dims)
    assert( (type(idx) == "number" and idx >= 0) or idx == "alloc", "expected an index number") -- alloc indicates that the solver should allocate the image as an intermediate
    self.P:Image(name,typ,ispace,idx,isunknown)
    local r = A.Image(name,self.P:ImageType(typ,ispace),not util.isvectortype(typ),isunknown and A.UnknownLocation or A.StateLocation)
    self.nametoimage[name] = r
    return r
end
function A.ProblemSpecAD:Unknown(name,typ,dims,idx) return self:Array(name,typ,dims,idx,true) end

function A.ProblemSpecAD:UnknownArgument(argpos)
    if not self.extraunknownarguments[argpos] then
        local r = {}
        for _,ip in ipairs(self.P:UnknownType().images) do
            local template = self.nametoimage[ip.name]
            r[ip.name] = A.Image(ip.name,template.type,template.scalar,A.ArgumentLocation(argpos))
        end
        self.extraunknownarguments[argpos] = r
    end
    return self.extraunknownarguments[argpos]
end

function A.ProblemSpecAD:ResidualArgument(argpos)
    if not self.extraresidualarguments[argpos] then
        local r = {}
        for _,rg in ipairs(self.P:ResidualType().groups) do
            local scalar = rg.imagetype.channelcount == 1
            r[rg.name] = A.Image(rg.name,rg.imagetype,scalar,A.ArgumentLocation(argpos))
        end
        self.extraresidualarguments[argpos] = r
    end
    return self.extraresidualarguments[argpos]
end

function A.IndexSpace:outerproduct(other)
    local dims = List()
    for _,d in ipairs(self.dims)    do dims:insert(d) end
    for _,d in ipairs(other.dims)   do dims:insert(d) end
    return A.IndexSpace(dims)
end

function A.ImageType:outerproduct(other)
    assert(A.ImageType:isclassof(other), "tried to outerproduct imagetype with "..tostring(other))
    assert(self.scalartype == other.scalartype)
    return A.ImageType(self.ispace:outerproduct(other.ispace), self.scalartype, self.channelcount*other.channelcount)
end

function A.ImageParam:outerproduct_name(other)
    assert(A.ImageParam:isclassof(other), "tried to outerproduct ImageParam name with "..tostring(other))
    return self.name.."_"..other.name
end

function A.ProblemSpecAD:JTJArgument()
    if not self.JTJ then
        local r = {}
        for _,ip0 in ipairs(self.P:UnknownType().images) do
            local template0 = self.nametoimage[ip0.name]
            r[ip0.name] = {}
            for _,ip1 in ipairs(self.P:UnknownType().images) do
                local template1 = self.nametoimage[ip1.name]
                local name = ip0:outerproduct_name(ip1)
                local imtype = template0.type:outerproduct(template0.type)
                local scalar = template0.scalar and template1.scalar
                r[ip0.name][ip1.name] = A.Image(name,imtype,scalar,A.JTJLocation)
            end
        end
        self.JTJ = r
    end
    return self.JTJ
end

function A.ProblemSpecAD:ImageTemporary(name,ispace,channelcount)
    local scalar = (channelcount == 1)
    local pixtyp = scalar and thallo_float or util.Vector(thallo_float,channelcount)
    self.P:Image(name,pixtyp,ispace,"alloc",false)
    local r = A.Image(name,self.P:ImageType(pixtyp,ispace),scalar,A.StateLocation)
    self.nametoimage[name] = r
    return r
end

function A.ProblemSpecAD:ImageWithName(name)
    return assert(self.nametoimage[name],"unknown image name?")
end

function A.Image:__tostring() return self.name end

function A.IterationDomain:ZeroOffset()
    if self._zerooffset then return self._zerooffset end
    local zeros = terralib.newlist()
    for i = 1,#self.domains do
        zeros:insert(0)
    end
    self._zerooffset = A.Offset(self.domains,zeros)
    return self._zerooffset
end




function A.IterationDomain:ZeroOffsetIndex()
    return A.ImageIndex(self.domains:map(util.wrap(A.DirectIndexComponent)))
end


local function constraintsforexpression(iteration_domain,exp)
    -- TODO(mmara): Re-enable using refactored API
    local usesbounds = false
    local bmin,bmax = iteration_domain:ZeroOffset(),iteration_domain:ZeroOffset()
    return A.BoundsAccess(bmin,bmax)
end


local function extract_unknowns(exp)
    local unknowns = terralib.newlist()
    local seen = {}
    local function addunknown(u)
        if not seen[u] then
            unknowns:insert(u)
            seen[u] = true
        end
    end
    local function visit_fn(a)
        if A.ImageAccess:isclassof(a) then
            if a.image.tensor_contraction then
                a.image.expression:visit(visit_fn) -- Recurse
            end
            if a.image.location == A.UnknownLocation then
                addunknown(a)
            elseif a.image.gradientimage then
                for i,unknown in ipairs(a.image.gradientimage.unknowns) do
                    local u = unknown:substitute(a.image.indexdomains,a.index.components)
                    addunknown(u)
                end
            end
        end
    end
    exp:visit(visit_fn)
    return unknowns
end

local function findUnknownsAndDomainsInExpression(exp)
    exp = assert(ad.toexp(exp),"expected a math expression")
    local domains_in_expression = {}
    exp:visit(function(a)
        if A.ImageAccess:isclassof(a) then
            assert(A.ImageIndex:isclassof(a.index))
            local new_domains = util.extract_domains_from_index(a.index)
            for _,d in ipairs(new_domains) do
                domains_in_expression[d] = true
            end
        elseif A.IndexValue:isclassof(a) then
            domains_in_expression[a.indexdomain] = true
        elseif A.SparseIndexValue:isclassof(a) then
            assert(A.ImageIndex:isclassof(a.access.index))
            local new_domains = util.extract_domains_from_index(a.access.index)
            for _,d in ipairs(new_domains) do
                domains_in_expression[d] = true
            end
        end
    end)
    return extract_unknowns(exp),domains_in_expression
end

function A.ProblemSpecAD:ComputedArray(name,indexdomains,exp)
    computed_array_count = computed_array_count + 1
    indexdomains = List(indexdomains)
    if ad.ExpVector:isclassof(exp) then
        local imgs = terralib.newlist()
        for i,e in ipairs(exp:expressions()) do
            imgs:insert(self:ComputedArray(name.."_"..tostring(i-1),indexdomains,e))
        end
        return A.ImageVector(imgs)
    end

    exp = assert(ad.toexp(exp),"expected a math expression")
    local unknowns, domains_in_expression = findUnknownsAndDomainsInExpression(exp)
    
    -- "Typechecking" to make sure the iteration domain fully cover the domain of the expression
    -- and vice-versa
    local passed_in_domains = {}
    for _,d in ipairs(indexdomains) do
        if not domains_in_expression[d] then
            print("Warning: Domain "..tostring(d).." not found in ComputeImage expression ")
        end
        passed_in_domains[d] = true
    end
    for d,_ in pairs(domains_in_expression) do 
        assert(passed_in_domains[d], "Domain "..tostring(d).." not found in ComputeImage domain list")
    end


    local dims = indexdomains:map(function(x) return x.dim end)
    local ispace = toispace(dims)
    local im = self:ImageTemporary(name,ispace,1)
    im.indexdomains = indexdomains
    
    if #unknowns > 0 then
        local gradients = exp:gradient(unknowns:map(function(x) return ad.v[x] end))
        local gim = self:ImageTemporary(name.."_gradient",ispace,#unknowns)
        gim.expressions = gradients
        gim.indexdomains = indexdomains
        im.gradientimage = A.GradientImage(unknowns,gim)
    end
    im.expression = exp
    local itdom = A.IterationDomain(indexdomains)
    im.constraints = constraintsforexpression(itdom,exp)
    self.precomputed:insert(A.PrecomputedDomain(im,itdom))
    return im
end

local function sort_index_domains(domains)
    table.sort(domains, function(d0,d1)
        return d0.index < d1.index
    end)
    return List(domains)
end

local function get_index_domains(exp)
    exp = assert(ad.toexp(exp),"expected a math expression")
    local domains = List()
    exp:visit(function(a)
        if A.ImageAccess:isclassof(a) then
            assert(A.ImageIndex:isclassof(a.index))
            local new_domains = util.extract_domains_from_index(a.index)
            domains:insertall(new_domains)
        end
    end)
    local sorted_domains = sort_index_domains(util.uniquify(domains))
    return sorted_domains
end

local function get_index_domains_from_exp_list(exp_list)
        local indexdomains = List()
        for _,e in ipairs(exp_list) do
            local new_domains = get_index_domains(e)
            indexdomains:insertall(new_domains)
        end
        indexdomains = sort_index_domains(util.uniquify(indexdomains))
        return indexdomains
end

local function get_index_domains_from_unknown_list(unknown_list)
    local domains = List()
    for _,u in ipairs(unknown_list) do
        assert(A.ImageAccess:isclassof(u))
        assert(A.ImageIndex:isclassof(u.index))
        local new_domains = util.extract_domains_from_index(u.index)
        domains:insertall(new_domains)
    end
    domains = sort_index_domains(util.uniquify(domains))
    return domains
end


local function createComputedArrayFromExp(ps,explike)
    local name = "StoredExp_"..tostring(computed_array_count)
    local indexdomains = List()
    if ad.ExpVector:isclassof(explike) then
        indexdomains = get_index_domains_from_exp_list(explike:expressions())
    else
        indexdomains = get_index_domains(explike)
    end
    return ps:ComputedArray(name,indexdomains,explike)
end

local function maybe_computed_array(explike,...)
    if not ComputedArrayCache[explike] then
        ComputedArrayCache[explike] = createComputedArrayFromExp(active_problem_spec,explike)
        ComputedArrayCache[explike].computed_array_index = computed_array_count
    end
    return ComputedArrayCache[explike](...)      
end

function ad.ExpVector:get(...)
    return maybe_computed_array(self,...)
end

function ad.Exp:get(...)
    return maybe_computed_array(self,...)
end

local function exp_set_materialize(explike, b)
    local materialization = ComputedArrayCache[explike]
    assert(materialization,"Currently can only materialize expressions gotten from")
    if A.ImageVector:isclassof(materialization) then
        for _,im in ipairs(materialization.images) do
            im:set_materialize(b)
        end
    else
        materialization:set_materialize(b)
    end
end

local function exp_set_gradient_materialize(explike, b)
    local materialization = ComputedArrayCache[explike]
    assert(materialization,"Currently can only materialize expressions gotten from")
    if A.ImageVector:isclassof(materialization) then
        for _,im in ipairs(materialization.images) do
            im:set_gradient_materialize(b)
        end
    else
        materialization:set_gradient_materialize(b)
    end
end

function ad.Exp:set_materialize(b)
    exp_set_materialize(self,b)
end
ad.ExpVector.set_materialize = ad.Exp.set_materialize

function ad.Exp:set_gradient_materialize(b)
    exp_set_gradient_materialize(self,b)
end
ad.ExpVector.set_gradient_materialize = ad.Exp.set_gradient_materialize

function A.Image:set_materialize(b)
    self.inline = not b
end

function A.Image:set_gradient_materialize(b)
    if self.gradientimage then
        self.gradientimage.image:set_materialize(b)
    else
        print("Warning: set_gradient_materialize on image without gradient")
    end
end

function A.Sparse:init() 
    self._is_coherent = false
end

function A.Sparse:set_coherent(b)
    self._is_coherent = b
    return self
end
function A.Sparse:__tostring() return self.name end
function A.ProblemSpecAD:Sparse(name,inspace,outspace,didx)
    self.P:Sparse(name,inspace,outspace,didx)
    return A.Sparse(name,toispace(inspace),toispace(outspace))
end

function A.ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,typ,idx)
    return A.ParamValue(name,typ):asvar()
end

local componentlikes = {"IndexDomain", "SparseAccess", "IndexComponent"}
local function iscomponentlike(candidate)
    return (type(candidate) == "number") or 
        any(function(class) return A[class]:isclassof(candidate) end, componentlikes)
end

local function insertIndexComponents(lst, args)
    if iscomponentlike(args) then
        lst:insert(tocomponent(args))
    else
        assert(type(args) == "table" and #args > 0) 
        for i=1,#args do
            insertIndexComponents(lst,args[i])
        end
    end
end

function A.Sparse:__call(first, ...)
    local index
    if A.ImageIndex:isclassof(first) then
        index = first
    else
        local components = terralib.newlist()
        insertIndexComponents(components,first)
        for i=1,select("#",...) do
            insertIndexComponents(components,select(i,...))
        end
        index = A.ImageIndex(components)
    end
    return A.SparseAccess(self, index)  
end


function A.Image:Exclude(exp)
    assert(self.location == A.UnknownLocation)
    exp = assert(ad.toexp(exp), "expected a AD expression")
    self.excludeexps:insert(exp)
end

function A.Image:DimCount() return #self.type.ispace.dims end
function A.Image:__call(first, ...)
    local index,c
    if A.ImageIndex:isclassof(first) then
        index = first
        c = select(1, ...)
    else
        local components = terralib.newlist()
        insertIndexComponents(components,first)
        for i=1,select("#",...) do
            if #components == self:DimCount() then
                c = select(i, ...)
            else
                insertIndexComponents(components,select(i,...))
            end
        end
        index = A.ImageIndex(components)
    end
    c = tonumber(c)
    assert(not c or c < self.type.channelcount, "Channel outside of range")
    if self.scalar or c then
        return A.ImageAccess(self,index,c or 0):asvar()
    else
        local r = {}
        for i = 1,self.type.channelcount do
            r[i] = A.ImageAccess(self,index,i-1):asvar()
        end
        return ad.Vector(unpack(r))
    end
end

-- Check if an index is trivially within bounds
function A.IndexSpace:index_trivially_in_bounds(index)
    local dims = self.dims
    local false_if_nontrivial = function(imIndex,offset)
        for i,c in ipairs(imIndex.components) do
            if A.DirectIndexComponent:isclassof(c) then
                if not (c.domain.dim == dims[i+offset]) then
                    return false
                end
            elseif not A.SparseIndexComponent:isclassof(c) then
                return false
            end
	   end
	   return true
    end
    local result = true
    if A.UnknownPairIndex:isclassof(index) then
        result = result and false_if_nontrivial(index.u0,0)
        result = result and false_if_nontrivial(index.u1,#index.u0.components)
    else
        assert(A.ImageIndex:isclassof(index))
        result = result and false_if_nontrivial(index,0)
    end
    return result
end

 -- wrapper for many images in a vector, just implements the __call method for Images Image:
function A.ImageVector:__call(...)
    local args = {...}
    local channelindex = self.images[1]:DimCount() + 1
    if #args == channelindex then
        local c = args[channelindex]
        assert(c < #self.images, "Channel outside of range")
        return self.images[c+1](unpack(args,1,channelindex-1))
    end
    local result = self.images:map(function(im) return im(unpack(args)) end)
    return ad.Vector(unpack(result))
end

function extract_simple_bounds(domain_like)
    if A.IndexDomain:isclassof(domain_like) then
        return domain_like,0
    elseif A.DirectIndexComponent:isclassof(domain_like) then
        return domain_like.domain,0
    else
        local err_message = "Your kludge has failed: InBounds() only works on indices of the form (domain (+/-) constant) or (constant + domain), but got "..tostring(domain_like)
        assert(A.ComponentBinOp:isclassof(domain_like), "b"..err_message)
        local directside = domain_like.lhs
        local constantside = domain_like.rhs
        if not A.DirectIndexComponent:isclassof(directside) or 
           not A.ConstantIndexComponent:isclassof(constantside) then
            constantside = domain_like.lhs
            directside = domain_like.rhs
            assert(domain_like.op == "+", err_message)
        end
        assert(A.DirectIndexComponent:isclassof(directside), "d"..err_message)
        assert(A.ConstantIndexComponent:isclassof(constantside), "c"..err_message)
        return directside.domain,constantside.value * (domain_like.op == "+" and 1 or -1)
    end
end

function thallo.InBounds(...)
    local args = {...}
    local offsets,domains = List(),List()
    for i,a in ipairs(args) do
        domains[i],offsets[i] = extract_simple_bounds(a)
    end
    local offset = A.Offset(domains,offsets)
    return A.BoundsAccess(offset,offset):asvar()
end
function thallo.InBoundsExpanded(...)
    local args = {...}
    local domains = List()
    local expand = args[#args]
    args[#args] = nil
    local min,max = List(),List()
    for i,a in ipairs(args) do
        assert(A.IndexDomain:isclassof(a))
        domains[i] = a
        min[i],max[i] = -expand, expand
    end
    return A.BoundsAccess(A.Offset(domains,min),A.Offset(domains,max)):asvar()
end
function A.BoundsAccess:type() return bool end --implementing AD's API for keys

function A.ResidualGroup:cardinality()
    return foldl(function(acc,next) return acc*next.dim.size end, 1, self:domain().domains)
end

function A.VarDef:shift(o) return self end
function A.BoundsAccess:shift(o)
    return A.BoundsAccess(self.min:shift(o),self.max:shift(o))
end
function A.ImageAccess:shift(o)
    return A.ImageAccess(self.image,self.index:shift(o),self.channel)
end

function A.Index:IsZero()
    for i,c in ipairs(self.components) do
        local zero_offset = not (A.DirectIndexComponent:isclassof(c) or A.SparseIndexComponent:isclassof(c))
        if zero_offset then return false end
    end
    return true
end

function A.Index:varies_with(index_domain)
    if A.UnknownPairIndex:isclassof(self) then
        return self.u0:varies_with(index_domain) or self.u1:varies_with(index_domain)
    else
        assert(A.ImageIndex:isclassof(self), "Unknown Index type")
        for i,c in ipairs(self.components) do
            local domains = util.extract_domains(c)
            if table.indexOf(domains,index_domain) >= 0 then
                if A.SparseIndexComponent:isclassof(c) then
                    return not c.access.sparse._is_coherent
                else
                    return true
                end
            end
        end
        return false
    end
end

function A.Offset:IsZero()
    for i,o in ipairs(self.data) do
        if o ~= 0 then return false end
    end
    return true
end

local function indexandoffset(component,domains,ignore_failure)
    assert(isnormalizedcomponent(component))
    local domain = util.extract_domain(component)
    local j = 0
    local off = 0
    for i,d in ipairs(domains) do
        if domain == d then
            j = i
            if A.ComponentBinOp:isclassof(component) then
                off = component.rhs.value
            end
        end
    end
    if not ignore_failure then
        assert(j>0, "Can't find image index component "..tostring(component).." in iteration domain!")
    end
    return j,off
end

-- Generic form for generating new offsets from two offsets or an offset/imageindex pair
function offsetfngenerator(self,rhs,binop)
    local r = List()
    for i = 1,#self.data do
        r[i] = self.data[i]
    end
    if A.ImageIndex:isclassof(rhs) then
        for _,comp in ipairs(rhs.components) do
            -- TODO: refactor this 
            local j,offset = indexandoffset(comp, self.domains, true)
            if offset ~= 0 and j ~= 0 then
                r[j] = binop(r[j],offset)
            end
        end
    else
        assert(A.Offset:isclassof(rhs), "Invalid type passed to Offset:Min()")
        if not rhs:IsZero() then
            for i = 1,#rhs.data do
                local j = table.indexOf(self.domains, rhs.domains[i])
                assert(j>0, "Can't find index domain "..tostring(rhs.domains[i]).." in iteration domain")
                r[j] = binop(r[j],rhs.data[i])
            end
        end
    end
    return A.Offset(self.domains, r)
end

function A.Offset:Min(rhs)
    return offsetfngenerator(self,rhs,math.min)
end
function A.Offset:Max(rhs)
    return offsetfngenerator(self,rhs,math.max)
end
function A.Offset:shift(rhs)
    return offsetfngenerator(self,rhs,function(x,y) return x+y end)
end

function A.Offset:__tostring()
    if self.str then return self.str end
    self.str = "("
    for i = 1,#self.data do
        local j = self.data[i]
        self.str = self.str..tostring(self.domains[i])
        if (j > 0) then
            self.str = self.str.."+"..tostring(j)
        elseif (j < 0) then
            self.str = self.str..tostring(j)
        end
        if i ~= #self.data then
            self.str = self.str..","
        end
    end
    self.str = self.str..")"
end


function A.IndexValue:get_component()
    return A.ComponentBinOp("+", A.DirectIndexComponent(self.indexdomain), A.ConstantIndexComponent(self._shift))
end

function A.IndexValue:shift(o)
    local ns = List()
    assert(A.ImageIndex:isclassof(o))
    local j = 0
    local off = 0
    for i,c in ipairs(o.components) do
        assert(isnormalizedcomponent(c))
        local domain = util.extract_domain(c)
        if domain == self.indexdomain then
            j = i
            if A.ComponentBinOp:isclassof(c) then
                off = c.rhs.value
            end
        end
    end
    assert(j>0, "Can't find image index domain "..tostring(self.indexdomain).." in iteration domain!")
    return A.IndexValue(self.indexdomain, self.shift_ + off)
end

function A.ImageIndex:shift(o)
    assert(A.ImageIndex:isclassof(o))
    local components = List()
    assert(#o.components == #self.components)
    for i,c in ipairs(o.components) do
        if A.DirectIndexComponent:isclassof(c) then
            components:insert(self.components[i])
        else
            assert(isnormalizedcomponent(c))
            components:insert(self.components[i]+c.rhs.value)
        end
    end
    return A.ImageIndex(components)
end

local function removeboundaries(exp)
    if ad.ExpVector:isclassof(exp) or terralib.islist(exp) then return exp:map(removeboundaries) end
    local function nobounds(a)
        if A.BoundsAccess:isclassof(a) and a.min:IsZero() and a.max:IsZero() then 
            return ad.toexp(1)
        else 
            return ad.v[a] 
        end
    end

    exp = exp:rename(nobounds)
    return exp
end

local nextirid = 0
function A.IRNode:init()
    self.id,nextirid = nextirid,nextirid+1
end
function A.Condition:create(members)
    local function cmp(a,b)
        if a.kind == "intrinsic" and b.kind ~= "intrinsic" then return true
        elseif a.kind ~= "intrinsic" and b.kind == "intrinsic" then return false
        else return a.id < b.id end
    end
    table.sort(members,cmp)
    return A.Condition(members)
end

function A.Condition:Intersect(rhs)
    local lhsmap = {}
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
    end
    local r = terralib.newlist()
    for i,m in ipairs(rhs.members) do
        if lhsmap[m] then
            r:insert(m)
        end
    end
    return A.Condition:create(r)
end

function A.Condition:Union(rhs)
    local lhsmap = {}
    local r = terralib.newlist()
    for i,m in ipairs(self.members) do
        lhsmap[m] = true
        r:insert(m)
    end
    for i,m in ipairs(rhs.members) do
        if not lhsmap[m] then
            r:insert(m)
        end
    end
    return A.Condition:create(r)
end

-- problemspec used only for types
local function createfunction(problemspec,name,fnkind,arguments,results,scatters,costonly)
    local Index = fnkind:indextype()

    local thread_count = Index.maxlinearindex

    local fnname = name.."_"..tostring(Index)..""
    results = removeboundaries(results)
    for i,s in ipairs(scatters) do
        s.expression = removeboundaries(s.expression)
    end
    local imageload = terralib.memoize(function(imageaccess)
        return A.vectorload(imageaccess,0,imageaccess.image.type:ElementType())
    end)
    local imagesample = terralib.memoize(function(image, x, y)
        return A.sampleimage(image,0,List{x,y},image.scalar and image.type.scalartype or image.type:ElementType())
    end)
    local imagesamplearray = terralib.memoize(function(image, x, y, z)
        return A.sampleimage(image,0,List{x,y,z},image.scalar and image.type.scalartype or image.type:ElementType())
    end)
    local irmap
    local function reshape_bounds_access(bounds)
        local mindata = List()
        local maxdata = List()
        local fdomains = fnkind.domain.domains
        for i=1,#fdomains do
            mindata[i] = 0
            maxdata[i] = 0
        end
        for i,domain in ipairs(bounds.min.domains) do
            local j = table.indexOf(fdomains, domain)
            if j > 0 then
                mindata[j] = bounds.min.data[i]
                maxdata[j] = bounds.max.data[i]
            end
        end
        return A.BoundsAccess(A.Offset(fdomains,mindata),A.Offset(fdomains,maxdata))
    end
    local function tofloat(ir,exp)
        if ir.type ~= thallo_float then
            return `thallo_float(exp)
        else
            return exp
        end    end
    local function createreduce(op,vardecl,n)
        local cond
        if op == "sum" and n.kind == "Apply" and n.op.name == "prod" then
            local conditions = terralib.newlist()
            local factors = terralib.newlist()
            for i,c in ipairs(n:children()) do
                if c:type() == bool then
                    conditions:insert(irmap(c))
                else
                    factors:insert(c)
                end
            end
            n = ad.prod(n.const,unpack(factors))
            cond = A.Condition:create(conditions)
        end
        return A.reduce(op,List{vardecl,irmap(n)},thallo_float,cond)
    end

    local function get_expression(expression, image, index)
        local function rename_fn(a)
            if A.ImageAccess:isclassof(a) or A.IndexValue:isclassof(a) then
                local newA = a:substitute(image.indexdomains,index.components)
                local result = ad.v[newA]
                return result
            elseif A.ParamValue:isclassof(a) then
                return ad.v[a]
            end
            return a
        end
        return expression:rename(rename_fn)
    end

    irmap = terralib.memoize(function(e)
        if ad.ExpVector:isclassof(e) then
            return A.vectorconstruct(e.data:map(irmap),util.Vector(thallo_float,#e.data))
        elseif "Var" == e.kind then
            local a = e:key()
            if "ImageAccess" == a.kind then
                if a.image.expression and (not a.image:Materialized()) then
                    return irmap(get_expression(a.image.expression, a.image, a.index))
                elseif a.image.expressions and (not a.image:Materialized(a.channel + 1)) then
                    return irmap(get_expression(a.image.expressions[a.channel + 1], a.image, a.index))
                end
                if not a.image.scalar then
                    local loadvec = imageload(A.ImageAccess(a.image,a.index,0))
                    loadvec.count = loadvec.count + 1
                    return A.vectorextract(List {loadvec}, a.channel, e:type())
                else
                    return A.load(a,e:type()) 
                end 
            else
                return A.intrinsic(a,e:type())
            end
        elseif "Const" == e.kind then
            return A.const(e.v,e:type())
        elseif "Apply" == e.kind then
            if (e.op.name == "sum") and #e:children() > 2 then
                local vardecl = A.vardecl(e.const,thallo_float)
                local children = List { vardecl }
                local varuse = A.varuse(children,thallo_float)
                for i,c in ipairs(e:children()) do
                    children:insert(createreduce(e.op.name,vardecl,c))
                end
                return varuse
            end
            local children = e:children():map(irmap)
            if e.op.name:match("^sampleimage") then
                local sm
                if e.op.name:match("^sampleimagearray") then
                    sm = imagesamplearray(e.op.imagebeingsampled,children[1],children[2],children[3])
                else
                    sm = imagesample(e.op.imagebeingsampled,children[1],children[2])
                end
                sm.count = sm.count + 1
                if sm.image.scalar then
                    return sm
                end
                return A.vectorextract(List {sm}, e.const, e:type()) 
            end
            local fn,gen = thallo.math[e.op.name]
            if fn then
                function gen(args)
                    local nargs = terralib.newlist()
                    for i,a in ipairs(args) do
                        nargs[i] = tofloat(children[i],a)
                    end
                    return `fn(nargs) 
                end
            else
                function gen(args) return e.op:generate(e,args) end
            end
            return A.apply(e.op.name,gen,children,e.const,e:type()) 
        end
    end)
    local irroots = results:map(irmap)
    for i,s in ipairs(scatters) do
        irroots:insert(irmap(s.expression))
    end
    
    local function linearizedorder(irroots)
        local visited = {}
        local linearized = terralib.newlist()
        local function visit(ir)
            if visited[ir] then return end
            visited[ir] = true
            if ir.children then
                for i,c in ipairs(ir.children) do visit(c) end
            end
            if ir.condition then
                for i,c in ipairs(ir.condition.members) do visit(c) end
            end
            linearized:insert(ir)
        end
        for i,r in ipairs(irroots) do
            visit(r)
        end
        return linearized
    end
    
    -- tighten the conditions under which ir nodes execute
    local linearized = linearizedorder(irroots)

    for i = #linearized,1,-1 do
        local ir = linearized[i]
        if not ir.condition then
            ir.condition = A.Condition:create(List{})
        end
        local function applyconditiontolist(condition,lst)
            for i,c in ipairs(lst) do
                if not c.condition then
                    c.condition = condition
                elseif c.kind == "reduce" then -- single use is this node, so the condition is the already established condition plus any that the variable use imposes
                    c.condition = c.condition:Union(condition)
                else
                    c.condition = c.condition:Intersect(condition)
                end
            end
        end
        if ir.children then applyconditiontolist(ir.condition,ir.children) end
        if ir.kind == "reduce" then applyconditiontolist(A.Condition:create(List{}), ir.condition.members) end
    end
    local function calculateusesanddeps(roots)
        local uses,deps = {},{}
        local function visit(parent,ir)
            if not deps[ir] then assert(not uses[ir])
                uses[ir],deps[ir] = terralib.newlist(),terralib.newlist()
                local function visitlist(lst)
                    for i,c in ipairs(lst) do
                        deps[ir]:insert(c)
                        visit(ir,c)
                    end
                end
                if ir.children then visitlist(ir.children) end
                if ir.condition then visitlist(ir.condition.members) end
            end
            if parent then
                uses[ir]:insert(parent)
            end
        end
        for i, r in ipairs(roots) do
            visit(nil,r)
        end
        return uses,deps
    end
    local uses,deps = calculateusesanddeps(irroots)

    local function prefixsize(a,b)
        for i = 1,math.huge do
            if a[i] ~= b[i] or a[i] == nil then return i - 1 end
        end
    end
    local function conditiondiff(current,next)
        local i = prefixsize(current.members,next.members)
        local uplevels,downlevels = #current.members - i, #next.members - i
        return uplevels,downlevels
    end
    local function conditioncost(current,next)
        local uplevels,downlevels = conditiondiff(current,next)
        return uplevels*1000 + downlevels
    end
        
    local function schedulebackwards(roots,uses)
        --print("Scheduling IR")
        local state = nil -- ir -> "ready" or ir -> "scheduled"
        local readylists = terralib.newlist()
        local currentcondition = A.Condition:create(List{})
        local function enter()
            state = setmetatable({}, {__index = state})
            readylists:insert(terralib.newlist())
        end
        enter() --initial root level for non-speculative moves
        
        for i,r in ipairs(roots) do
            if not state[r] then -- roots may appear in list more than once
                -- It is possible for a member of the irroots list to
                -- not actually be a root of the DAG; prune those out
                if #uses[r] == 0 then
                    state[r] = "ready"
                    readylists[#readylists]:insert(r)
                end
            end
        end
        
        local function registersreleased(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" or ir.kind == "sampleimage" then return ir.count
            elseif ir.kind == "vectorextract" then return 0
            elseif ir.kind == "varuse" then return 0
            elseif ir.kind == "vardecl" then return 1
            elseif ir.kind == "reduce" then return 0 
            else return 1 end
        end
        local function registersliveonuse(ir)
            if ir.kind == "const" then return 0
            elseif ir.kind == "vectorload" then return 0
            elseif ir.kind == "sampleimage" then return util.isvectortype(ir.type) and 0 or 1
            elseif ir.kind == "vectorextract" then return 1
            elseif ir.kind == "varuse" then return 1
            elseif ir.kind == "reduce" then return 0
            elseif ir.kind == "vardecl" then return 0
            else return 1 end
        end
        local function netregisterswhenscheduled(ir)
            local n = -registersreleased(ir)
            for _,k in ipairs(deps[ir]) do
                if not state[k] then
                    n = n + registersliveonuse(k)
                end
            end
            return n
        end
        local function checkandmarkready(ir)
            if state[ir] ~= "ready" then
                for i,u in ipairs(uses[ir]) do
                    if state[u] ~= "scheduled" then return end -- not ready
                end            
                readylists[#readylists]:insert(ir)
                state[ir] = "ready"
            end
        end
        local function markscheduled(ir)
            state[ir] = "scheduled"
            for i,c in ipairs(deps[ir]) do 
                if not state[c] then
                    state[c] = "used"
                end
                checkandmarkready(c)
            end
        end
        
        local function vardeclcost(ir)
            return ir.kind == "vardecl" and 0 or 1
        end

        local function cost(idx,ir)
            local c =  { 0 }
            table.insert(c, conditioncost(currentcondition,ir.condition))
            table.insert(c, vardeclcost(ir))
            table.insert(c, netregisterswhenscheduled(ir))
            return c
        end
        
        local function costless(a,b)
            for i,ac in ipairs(a) do
                local bc = b[i]
                if ac ~= bc then return ac < bc end
            end
            return false
        end
        -- TODO: maintain costs, update incrementally? This is plurality of Thallo-controllable compile time...
        local ready = readylists[1] -- the true ready list is the first one, the rest are the speculative lists
        local function choose()
            --print("---------------------")
            local best = cost(1,assert(ready[1]))
            local bestidx = 1
            for i = 2,#ready do
                local ci = cost(i,ready[i])
                if costless(ci,best) then
                    bestidx = i
                    best = ci
                end
            end
            --print("choose",bestidx)
            return table.remove(ready,bestidx)
        end
        
        local instructions = terralib.newlist()
        local regcounts = terralib.newlist()
        local currentregcount = 1
        while #ready > 0 do
            local ir = choose()
            instructions:insert(1,ir)
            regcounts:insert(1,currentregcount)
            currentregcount = currentregcount + netregisterswhenscheduled(ir)
            markscheduled(ir)
            currentcondition = ir.condition
        end
        --print("Scheduled IR")
        return instructions,regcounts
    end

    local instructions,regcounts = schedulebackwards(irroots,uses)
    
    local function printschedule(W,instructions,regcounts)
        W:write(string.format("schedule for %s -----------\n",name))
        local emittedpos = {}
        local function formatchildren(children)
            local cs = terralib.newlist()
            for i,c in ipairs(children) do
                cs:insert("r"..tostring(emittedpos[c]))
            end
            return cs:concat(",")
        end
    
        local function formatinst(inst)
            local fs = terralib.newlist()
            fs:insert(inst.kind.." ")
            for k,v in pairs(inst) do
                if k ~= "kind" and k ~= "children" and type(v) ~= "function" and k ~= "id" and k ~= "condition" and k ~= "type" then
                    fs:insert(tostring(v))
                    fs:insert(" ")
                end
            end
            if inst.children then
                fs:insert("{")
                fs:insert(formatchildren(inst.children))
                fs:insert("}")
            end
            return fs:concat()
        end
        local function formatcondition(c)
            local fs = terralib.newlist()
            fs:insert("[")
            fs:insert(formatchildren(c.members))
            fs:insert("]")
            local r = fs:concat()
            return r .. (" "):rep(4*(1+#c.members) - #r)
        end
        for i,ir in ipairs(instructions) do
            emittedpos[ir] = i
            W:write(("[%d]%sr%d : %s = %s\n"):format(regcounts[i],formatcondition(ir.condition),i,tostring(ir.type),formatinst(ir)))
            if instructions[i+1] and conditioncost(ir.condition,instructions[i+1].condition) ~= 0 then
                W:write("---------------------\n")
            end
        end
        W:write("----------------------\n")
    end
    
    if verboseAD then
        local W = io.open("log.txt","a")
        printschedule(W,instructions,regcounts)
        W:close()
    end
    
    local P = symbol(problemspec:ParameterType(),"P")

    local ridx = symbol(Index,"ridx")
    
    local statementstack = terralib.newlist { terralib.newlist() } 
    local statements = statementstack[1]
    local TUnknownType = problemspec:UnknownType():terratype()
    local TResidualType = problemspec:ResidualType():terratype()
    local TJTJType = problemspec:JTJType():terratype()
    local extraarguments = arguments:map(function(a) 
        if A.UnknownArg:isclassof(a) then
            return symbol(TUnknownType,a.name) 
        elseif A.ResidualArg:isclassof(a) then
            return symbol(TResidualType,a.name) 
        elseif A.UnknownPairArg:isclassof(a) then
            return symbol(TJTJType,a.name) 
        else
            assert(false, "invalid extra argument type")
        end
    end)
    local emit
    local function emitconditionchange(current,next)
        local u,d = conditiondiff(current,next)
        for i = 0,u - 1 do
            local c = current.members[#current.members - i]
            local ce = emit(c)
            local stmts = statementstack:remove()
            statementstack[#statementstack]:insert quote
                if ce then
                    [stmts]
                end
            end
        end
        for i = 1,d do
            statementstack:insert(terralib.newlist())
        end
        statements = statementstack[#statementstack]
    end
    local function get_domain_index(domain)
        local j = table.indexOf(fnkind.domain.domains, domain)
        assert(j>0, "Can't find index domain "..tostring(domain).." in fnkind domain: "..tostring(fnkind.domain))
        return j
    end
    local function boundcoversload(ba,index)
        ba = reshape_bounds_access(ba)
        for _,comp in ipairs(index.components) do
            assert(A.DirectIndexComponent:isclassof(comp), "Error: Trying to check bounds for sparse access.")
            local j = get_domain_index(comp.domain)
            local o,bmin,bmax = 0,ba.min.data[j],ba.max.data[j]
            if o < bmin or o > bmax then
                return false
            end
        end
        return true
    end
    local function conditioncoversload(condition,index)
        for _,comp in ipairs(index.components) do
            if (not A.DirectIndexComponent:isclassof(comp)) and (not A.SparseIndexComponent:isclassof(comp)) then
                return false
            end
        end
        return true 
    end
    local function imageref(image)
        if image.location == A.StateLocation then
            return `P.[image.name]
        elseif image.location == A.UnknownLocation then
            return `P.X.[image.name]
        elseif image.location == A.JTJLocation then
            local sym = assert(extraarguments[#extraarguments],"JTJ image not found")
            return `sym.[image.name]
        else
            local sym = assert(extraarguments[image.location.idx],"Unknown extra image")
            return `sym.[image.name]
        end
    end

    local getIndex --forward declaration
    local index_cache = {}
    local function getindirectindexvalue(access, statements, condition)
        local sparse = access.sparse
        local sparseInd = access.index
        local edIndex = getIndex(sparse.inspace:indextype(),sparseInd,statements,condition)
        if print_sparse_indices then
           statements:insert( quote printf([sparse.name.."[%d]\n"], [edIndex:tooffset()]) end )
        end
        if stub_out_sparse_loads then
            return `0
        end
        return `P.[sparse.name][edIndex:tooffset()].["d0"]
    end

    local function get_index_component_value(component, statements, condition)
        local function recurse(c) return get_index_component_value(c, statements, condition) end
        local c = component
        if A.DirectIndexComponent:isclassof(c) then
            local domain = c.domain
            local j = get_domain_index(domain)
            return `ridx.["d"..tostring(j-1)]
        elseif A.SparseIndexComponent:isclassof(c) then
            return getindirectindexvalue(c.access, statements, condition)
        elseif A.ConstantIndexComponent:isclassof(c) then
            return `c.value
        else
            assert(A.ComponentBinOp:isclassof(c), "Unknown index component type!")
            if c.op == "+" then
                return `[recurse(c.lhs)] + [recurse(c.rhs)]
            else
                assert(c.op == "-", "Unknown index op")
                return `[recurse(c.lhs)] - [recurse(c.rhs)]
            end
        end
    end

    -- TODO: Allow nesting? Don't emit redundant sparse accesses? Have all accesses go through here?
    getIndex = function (indextype,ind,statements,condition)
        if not condition then condition = A.Condition(List()) end
        local cache_i = A.ConditionIndexPair(condition,ind)
        local index = index_cache[cache_i] 
        if not index then
            local imIndexType = indextype
            index = symbol(imIndexType,tostring(ind))
            --print("Inserting to cache: "..tostring(index))
            statements:insert( quote var [index] end )
            for i=1,#ind.components do
                local value = get_index_component_value(ind.components[i], statements, condition)
                statements:insert( quote [index].["d"..tostring(i-1)] = value end )
            end
        end
        index_cache[cache_i] = index
        return index
    end


    -- TODO: don't rely on subsequent optimization passes to do this efficiently
    -- Probably want to linearize the components so its a 2-component index
    -- This is in scatters only, so no need for conditions in the cache index
    local function getProductIndex(indextype,ind,statements, condition)
        if not condition then condition = A.Condition(List()) end
        assert(A.UnknownPairIndex:isclassof(ind))
        local imIndex = index_cache[ind] 
        if not imIndex then
            local imIndexType = indextype
            imIndex = symbol(imIndexType,tostring(ind))
            statements:insert( quote var [imIndex] end )
            local u0comps = ind.u0.components
            local u1comps = ind.u1.components
            for i=1,#u0comps do
                local c = u0comps[i]
                local value = get_index_component_value(c, statements, condition)
                statements:insert( quote [imIndex].["d"..tostring(i-1)] = value end )
            end
            for k=1,#u1comps do
                local c = u1comps[k]
                local i = k + #u0comps
                local value = get_index_component_value(c, statements, condition)
                statements:insert( quote [imIndex].["d"..tostring(i-1)] = value end )
            end
        end
        index_cache[ind] = imIndex
        return imIndex
    end

    local declarations = terralib.newlist()

    local function createexp(ir)        
        if "const" == ir.kind then
            return `thallo_float(ir.value)
        elseif "intrinsic" == ir.kind then
            local a = ir.value
            if "BoundsAccess" == a.kind then--bounds calculation
                -- Reshape Bounds Access
                local reshaped_bounds = reshape_bounds_access(a)
                return `ridx:InBoundsExpanded([reshaped_bounds.min.data],[reshaped_bounds.max.data])
            elseif "IndexValue" == a.kind then
                local j = get_domain_index(a.indexdomain)
                local n = "d"..tostring(j-1)
                return `ridx.[n] + a.shift_ 
            elseif "SparseIndexValue" == a.kind then
                return getindirectindexvalue(a.access, statements,ir.condition)
            else assert("ParamValue" == a.kind)
                if print_param_access then statements:insert(quote printf(["ParamValue("..a.name..")\n"]) end) end
                if stub_out_param_access then
                    return `thallo_float(0.0)
                end
                return `thallo_float(P.[a.name])
            end
        elseif "load" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            local imIndex = getIndex(a.image:indextype(),a.index,declarations,ir.condition)
            if conditioncoversload(ir.condition,a.index) then
               return `im(imIndex)(0) 
            else
               return `im:get(imIndex)(0)
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            local im = imageref(a.image)
            local s = symbol(a.image.type:ElementType(),("%s_%s"):format(a.image.name,tostring(a.index)))
            local imIndex = getIndex(a.image:indextype(),a.index,declarations, ir.condition)
            if conditioncoversload(ir.condition,a.index) then
                statements:insert(quote
                    var [s] = im(imIndex)
                end)
            else 
                statements:insert(quote
                    var [s] = 0.f
                    if imIndex:InBounds() then
                        [s] = im(imIndex)
                    end
                end)
            end

            return s
        elseif "vectorextract" == ir.kind then
            local v = emit(ir.children[1])
            return `v(ir.channel)
        elseif "vectorconstruct" == ir.kind then
            local exps = ir.children:map(emit)
            return `[util.Vector(thallo_float,#exps)]{ array(exps) }
        elseif "sampleimage" == ir.kind then -- 2D or 2Darray
            local im = imageref(ir.image)
            local exps = ir.children:map(emit)
            local r = `im:sample(exps)
            if ir.image.scalar then
                r = `r(0)
            end
            return r
        elseif "apply" == ir.kind then
            local exps = ir.children:map(emit)
            return ir.generator(exps)
        elseif "vardecl" == ir.kind then
            return `thallo_float(ir.constant)
        elseif "varuse" == ir.kind then
            local children = ir.children:map(emit)
            return children[1] -- return the variable declaration, which is the first child
        elseif "reduce" == ir.kind then
            local children = ir.children:map(emit)
            local vd, exp = children[1], tofloat(ir.children[2],children[2])
            local op
            if ir.op == "sum" then
                op = quote [vd] = [vd] + [exp] end
            else
                op = quote [vd] = [vd] * [exp] end
            end
            statements:insert(op)
            return children[1]
        end
    end
    
    local emitted,emitteduse = {},{}
    
    function emit(ir)
        assert(ir)
        return assert(emitted[ir],"Use before def")
    end

    local basecondition = A.Condition:create(List{})
    local currentcondition = basecondition
    
    local rsymbols = terralib.newlist()
    local rsymbolmap = {}
    for i,ir in ipairs(instructions) do
        emitconditionchange(currentcondition,ir.condition)
        currentcondition = ir.condition
        
        if false then -- dynamically check dependencies are initialized before use, very slow, only use for debugging
            local ruse = symbol(bool,"ruse"..tostring(i))
            declarations:insert quote var [ruse] = false end
            statements:insert quote [ruse] = true end
            emitteduse[ir] = ruse
            for _,u in ipairs(deps[ir]) do
                if ir.kind ~= "varuse" or ir.children[1] == u then
                    local ruse = assert(emitteduse[u])
                    local str = ("%s r%s used %s which is not initialized\n"):format(name,tostring(i),tostring(ruse))
                    statements:insert quote
                        if not ruse then
                            printf(str)
                        end
                    end
                end
            end
        end
        
        local r
        if ir.kind == "const" or ir.kind == "varuse" or ir.kind == "reduce" then 
            r = assert(createexp(ir),"nil exp") 
        else
            r = symbol(ir.type,"r"..tostring(i))
            rsymbols:insert(r)
            rsymbolmap[i] = #rsymbols
            declarations:insert quote var [r] end
            local exp = assert(createexp(ir),"nil exp")
            statements:insert(quote
                [r] = exp;
            end)
        end
        emitted[ir] = r
    end


    --[[
    Based on 5.4.1. Arithmetic Instructions in the 
    Cuda C Programming Guide
    https://docs.nvidia.com/cuda/archive/10.1/pdf/CUDA_C_Programming_Guide.pdf

    These have 1/4 the throughput of simple ops (add/sub/mul) when implemented in intrinsics

    32-bit floating point reciprocal,
    reciprocal square root,
    base-2 logarithm (__log2f), 
    base 2 exponential (exp2f), 
    sine (__sinf), 
    cosine (__cosf)

    Tangent: Derived from its implementation as __sinf(x) * (1/__cosf(x)) (12 ops)
    __powf(x, y): Derived from its implementation as exp2f(y *__log2f(x)). (9 ops)

    compare/min/max are half the throughput of simple ops

    The rest are currently heuristic estimates
    --]]

    local mathopcount = {
        sqrt = 4,
        cos  = 4,
        acos = 8,
        sin  = 4,
        asin = 8,
        tan  = 12,
        atan = 16,
        ceil = 2,
        floor = 2,
        log = 4,
        exp = 4,
        pow  = 4,
        fmod = 2,
        fmax = 2,
        fmin = 2,
        greatereq = 2,
        greater = 2,
        greatereq = 2,
        less = 2,
        lesseq = 2,
        eq = 2,
        neq = 2,
        not_ = 1,
        and_ = 2,
        or_ = 2,
        abs = 1
    }


    local body_ops = 0
    local body_reads = 0


    -- Returns both if the value is coalesced and further if its constant w.r.t. the inner iterator
    -- TODO: Elide inner iterator if unit
    local function is_coalesced(index,fnkind_domain)
        --print(index)
        --print(fnkind_domain)
        if A.ImageIndex:isclassof(index) then
            local all_domains = util.extract_domains_from_index(index)
            local first_component_domains =  util.extract_domains(index.components[1])

            local function is_inner_iterator(d)
                return fnkind_domain.domains[1] == d
            end

            if util.exists(all_domains, is_inner_iterator) then
                if util.exists(first_component_domains, is_inner_iterator) then
                    if #first_component_domains == 1 then
                        --print("Coalesced")
                        return true,false
                    else
                        --print("Coalesced with multiple domains")
                        return true,false
                    end
                else
                    -- TODO: handle constant?
                    --print("Uncoalesced")
                    return false,false
                end
            else
                --print("Beyond coalesced; all threads in warp read same?")
                return true,true
            end
            assert(false)
            return false,false
        end
        assert(false)
        return false,false
    end

    local function coalescence_multiplier(index,fnkind_domain)
        local coalesced,not_varying = is_coalesced(index, fnkind_domain)
        if coalesced then
            return not_varying and (1.0/util.cu.transactions_per_coalesced_read) or 1
        end
        return  util.cu.uncoalesced_multiplier
    end

    
    for i,ir in ipairs(instructions) do
        if "const" == ir.kind then
            -- Nothing
        elseif "intrinsic" == ir.kind then
            local a = ir.value
            if "BoundsAccess" == a.kind then--bounds calculation
                local dimcount = Index.ispace:DimCount()
                local ops = 4 * dimcount - 1
                body_ops = body_ops + ops
            elseif "IndexValue" == a.kind then
                body_ops = body_ops + 1
            elseif "SparseIndexValue" == a.kind then
                body_reads = body_reads + 1
            else assert("ParamValue" == a.kind)
                -- Nothing
            end
        elseif "load" == ir.kind then
            local a = ir.value
            body_reads = body_reads + 1 * coalescence_multiplier(a.index,fnkind.domain)
            if not conditioncoversload(ir.condition,a.index) then
                body_ops = body_ops + a.image:DimCount() * 3 - 1
            end
        elseif "vectorload" == ir.kind then
            local a = ir.value
            body_reads = body_reads + (a.image.type.channelcount * coalescence_multiplier(a.index,fnkind.domain))
            if not conditioncoversload(ir.condition,a.index) then
                body_ops = body_ops + a.image:DimCount() * 3 - 1
            end
        elseif "vectorextract" == ir.kind then
            --nothing
        elseif "vectorconstruct" == ir.kind then
            -- nothing
        elseif "sampleimage" == ir.kind then -- 2D or 2Darray
            local im = imageref(ir.image)

            local v = ir.image.type.channelcount
            if 2 == ir.image:DimCount() then
                --[[
                local terra lerp(v0 : vectortype, v1 : vectortype, t : thallo_float)
                    return (thallo_float(1.) - t)*v0 + t*v1 -- 1 + 3*v ops
                end

                terra Image:sample(x : thallo_float, y : thallo_float)
                    var x0 : int, x1 : int = thallo.math.floor(x),thallo.math.ceil(x) -- 2 ops
                    var y0 : int, y1 : int = thallo.math.floor(y),thallo.math.ceil(y) -- 2 ops
                    var xn,yn = x - x0,y - y0 -- 2 ops
                    var u = lerp(self:get( Index {x0,y0} ),self:get( Index {x1,y0} ),xn) -- lerp + 2 gets
                    var b = lerp(self:get( Index {x0,y1} ),self:get( Index {x1,y1} ),xn) -- lerp + 2 gets
                    return lerp(u,b,yn) -- lerp + 2 gets
                end
                total 6 ops + 3 lerps + 4 gets
                --]]
                local lerp_ops = v * 3 + 1
                local get_ops = ir.image:DimCount() * 3 - 1
                body_ops = body_ops + 6 + 3*lerp_ops + 4*get_ops
                body_reads = body_reads + 4 * v
            elseif 3 == ir.image:DimCount() then
                --[[
                terra Image:horizontalConditionalLerp(s : &vectortype, w : &float, x : int, y : int, z : int, alpha : float, imageWidth : int, imageHeight : int)
                    if (x >= 0 and y >= 0 and x < imageWidth and y < imageHeight) then -- 2ops
                        var v : vectortype = self(Index {x,y,z}) -- v reads
                        if (v(0) ~= [-math.huge]) then  -- 1 op
                            @s = @s + (alpha*v); -- 2 v ops
                            @w = @w + alpha;  -- v ops
                        end 
                    end
                end
                3v + 3 ops
                v reads

                terra Image:sample(x : thallo_float, y : thallo_float, z : int)
                    var imageWidth : int = [imispace.dims[1].size]
                    var imageHeight : int = [imispace.dims[2].size]
                    var x0 : int, x1 : int = thallo.math.floor(x),thallo.math.ceil(x) -- 2 ops
                    var y0 : int, y1 : int = thallo.math.floor(y),thallo.math.ceil(y) -- 2 ops
                    var alpha = x - x0 -- 1 op
                    var beta = y - y0 -- 1 op

                    var s0 : vectortype = 0.0f
                    var w0 : float = 0.0f
                    self:horizontalConditionalLerp(&s0, &w0, x0, y0, z, (1.0f - alpha), imageWidth, imageHeight) -- 1 op, 1 hcl
                    self:horizontalConditionalLerp(&s0, &w0, x1, y0, z, alpha, imageWidth, imageHeight) --  1 hcl
                    
                    var s1 : vectortype = 0.0f
                    var w1 : float = 0.0f
                    self:horizontalConditionalLerp(&s1, &w1, x0, y1, z, (1.0f - alpha), imageWidth, imageHeight) --  1 hcl
                    self:horizontalConditionalLerp(&s1, &w1, x1, y1, z, alpha, imageWidth, imageHeight) --  1 hcl

                    var p0 = s0 * (1.0f/w0) -- v + 4 op
                    var p1 = s1 * (1.0f/w1) -- v + 4 op

                    var ss : vectortype = 0.0f 
                    var ww : float = 0.0f
                    if (w0 > 0.0f) then -- 1 op
                        ss = ss + (1.0f - beta)*p0 -- 3 op
                        ww = ww + (1.0f - beta) -- 1 op
                    end
                    if (w1 > 0.0f) then  -- 1 op
                        ss = ss + beta * p1 -- 2 v op
                        ww = ww + beta -- 1 op
                    end

                    if (ww > 0.0f) then  -- 1 op
                        return ss / ww -- 4v op
                    else
                        return vectortype([-math.huge])
                    end
                end
                -- total 22 op + 8 vop + 4hcl = 34 ops + 20 vops

                --]]
                body_ops = body_ops + 34 + 20*v 
                body_reads = body_reads + 4*v
            else
                assert(false, "NYI: sample for dimensions besides 2d and 3d")
            end
        elseif "apply" == ir.kind then
            assert(ir.type.name == "float" or ir.type.name == "bool")
            local op_count = mathopcount[ir.op] 
            if op_count ~= nil then
                body_ops = body_ops + op_count
            elseif ir.op == "powc" then
                if ir.const < 0 then
                    body_ops = body_ops + 4 + (-ir.const-1)
                else
                    body_ops = body_ops + ir.const-1
                end
            elseif ir.op == "prod" then
                if ir.const and not (ir.const == 1 or ir.const == -1) then
                    body_ops = body_ops + 1 
                end
                body_ops = body_ops + (#ir.children - 1)
            elseif ir.op == "sum" then
                if ir.const and not (ir.const == 0) then
                    body_ops = body_ops + 1
                end
                body_ops = body_ops + (#ir.children - 1)
            elseif ir.op == "select" then
                body_ops = body_ops + 1
            elseif ir.op == "constant" then
                -- Nothing
            else
                print(ir.type.name)
                print(ir.op)
                print(#ir.children)
                assert(false, "op not implemented in cost evaluation")
            end
        elseif "vardecl" == ir.kind then
            -- Nothing
        elseif "varuse" == ir.kind then
            -- Nothing
        elseif "reduce" == ir.kind then
            assert(#ir.children == 2 and (ir.type.name == "float" or ir.type.name == "bool"))
            body_ops = body_ops + #ir.children - 1
        end
    end
    --print("Ops: "..tostring(body_ops))
    --print("Reads: "..tostring(body_reads))
  
    
    emitconditionchange(currentcondition,A.Condition:create(List{}))
    assert(#statementstack == 1)
    
    local expressions = irroots:map(emit)
    local resultexpressions,scatterexpressions = {unpack(expressions,1,#results)},{unpack(expressions,#results+1)}
        
    local scatterstatements = terralib.newlist()
    local function toidx(p,statements)
        if A.UnknownPairIndex:isclassof(p.index) then
            return getProductIndex(p.space:indextype(),p.index,statements)
        else
            return getIndex(p.space:indextype(),p.index,statements)
        end
    end
    --[[
    print("ALL SCATTERS "..name)
    for i,s in ipairs(scatters) do
        print("Scatter "..tostring(i)..": "..tostring(s.image).."["..tostring(s.index).."]("..tostring(s.channel)..")")
    end
    --]]

    local function maybe_insert_debugging_printf(scatterstatements,exp,ridx)
        if false and (name == "evalJTF" or name == "applyJ" or name == "applyJt") and A.ResidualAndContractionwiseFunction:isclassof(fnkind) then
            scatterstatements:insert quote
                if ridx.d0 % 10000 == 0 and ridx.d1 % 10000 == 0 then
                    printf([name.." %d,%d: %f\n"], [int32](ridx.d0), [int32](ridx.d1), [thallo_float](exp))
                end
            end
        end
    end

    -- Clear index cache for scatters
    index_cache = {}

    local ISpairs,ISpairToScatters = MapAndGroupByI(scatters, function(i,s)
        return A.IndexAndSpacePair(s.index,s.image.type.ispace), i
    end)

    local indexToPeers = {}

    local scatter_regs = 0
    local scatter_ops = 0
    local scatter_adds = 0
    local scatter_writes = 0
    -- TODO: compute register counts for all bounds checks
    -- TODO: compute op counts for index expressions
    for i,p in ipairs(ISpairs) do
        local scats = ISpairToScatters[p]
        local skind = scatters[scats[1]].kind
        for _,ind in ipairs(scats) do
            assert(scatters[i].kind == skind, "All scatters to the same image domain must be the same type")
        end

        local index = toidx(p,scatterstatements)
        local trivially_in_bounds = p.space:index_trivially_in_bounds(p.index)
        local inner_index_domain = fnkind.domain.domains[1]
        local varying_with_inner_iterate = p.index:varies_with(inner_index_domain)
        local stmt
        if skind == "add" then
            --print(tostring(p.index).." trivially within bounds? "..tostring(trivially_in_bounds))
            --print(tostring(#scats).." scatters")
            local inbounds = `true
            if not trivially_in_bounds then
                inbounds = symbol(bool,"inbounds")
                scatterstatements:insert(quote var [inbounds] = index:InBounds() end)
                scatter_ops = scatter_ops + p.space:DimCount()
            end
            if varying_with_inner_iterate then
                stmt = quote if inbounds then
                    escape
                        for _,ind in ipairs(scats) do
                            local s = scatters[ind]
                            local image,exp = imageref(s.image),scatterexpressions[ind]
                            maybe_insert_debugging_printf(scatterstatements,exp,ridx)
                            if index.type.ispace._string == "C_C" then
                                -- TODO(mmara): Debug bundle adjustment JtJ (multiple unknown domains)
                                --util.rPrint(index,4)
                                --util.rPrint(s.image.indextype,4)
                            end
                            emit `image:atomicAddChannel(index, s.channel, [thallo_float](exp)) 
                        end
                    end 
                end end
                scatter_adds = scatter_adds + #scats
                scatterstatements:insert(stmt)
            else
                local peers = indexToPeers[index]
                if not peers then
                    peers = symbol(uint,"peers"..tostring(index))
                    scatterstatements:insert(quote var [peers] = util.get_peers(index:tooffset()) end)
                    if guard_atomics and not trivially_in_bounds then
                        scatterstatements:insert(quote 
                            if not inbounds then
                                [peers] = 0
                            end
                        end)
                    end
                    indexToPeers[index] = peers
                    -- TODO: accurate opcount for get_peers
                    scatter_ops = scatter_ops + 6*5
                end
                for _,ind in ipairs(scats) do
                    local s = scatters[ind]
                    local image,exp = imageref(s.image),scatterexpressions[ind]
                    maybe_insert_debugging_printf(scatterstatements,exp,ridx)
                    stmt = `image:aggregatedAtomicAddChannel(index, s.channel, [thallo_float](exp), [peers])
                    scatterstatements:insert(stmt)
                    scatter_adds = (scatter_adds*util.cu.uncoalesced_multiplier) / util.cu.warpSize
                end
            end
        else
            for _,ind in ipairs(scats) do
                local s = scatters[ind]
                local image,exp = imageref(s.image),scatterexpressions[ind]
                assert(skind == "set")
                local stmt = quote 
                    image:setChannel(index, s.channel, [thallo_float](exp))
                end
                scatterstatements:insert(stmt)

                scatter_writes = scatter_writes + (1 * coalescence_multiplier(p.index,fnkind.domain))
            end
        end
        
    end

    if name == "cost" then 
        --statements = splicereturnstatement(statements,107,{59,104}) 
    end
    --print("Declaring terra function "..name)

    -- TODO: empirically compare?
    local register_count = max(regcounts) + 5 + scatter_regs
    local opcount_per_thread = body_ops + scatter_ops + scatter_adds
    local memreads_per_thread = body_reads + scatter_adds
    local memwrites_per_thread = scatter_writes + scatter_adds
    local costdata = A.KernelCostData(thread_count, register_count, opcount_per_thread, memreads_per_thread, memwrites_per_thread)
    --[[if name == "materializeJTJ" or name == "evalJTF" then
        print(name)
        print(min(regcounts))
        print(max(regcounts))
        print(max(regcounts)-min(regcounts))
        print(costdata)
    end--]]
    local generatedfn
    if not costonly then
        local idx = symbol(Index,"idx")
        generatedfn = terra ([idx], [P], [extraarguments])
            var [ridx] = idx
            [declarations]
            [statements]
            [scatterstatements]
            return [resultexpressions]
        end
        generatedfn:setname(name)
        if verboseAD then
            generatedfn:printpretty(false, false)
        end
    end
    
    return generatedfn, costdata
end

local noscatters = terralib.newlist()

function compilefunctionspec(problemspec, functionspec)
    return createfunction(problemspec,functionspec.name,functionspec.kind,functionspec.arguments,functionspec.results,functionspec.scatters)
end

function getfunctioncostdata(problemspec,functionspec)
    local _, cost_data = createfunction(problemspec,functionspec.name,functionspec.kind,functionspec.arguments,functionspec.results,functionspec.scatters,true)
    return cost_data
end

function A.ProblemSpecAD:CompileFunctionSpec(functionspec)
    return compilefunctionspec(self.P,functionspec)
end

function A.ProblemSpecAD:AddFunctions(functionspecs)
    local kind_to_functionmap = {}
    local kinds = List()
    for i,fs in ipairs(functionspecs) do -- group by unique function kind to pass to ProblemSpec:Functions call
        local fm = kind_to_functionmap[fs.kind]
        if not fm then
            fm = {}
            kind_to_functionmap[fs.kind] = fm
            kinds:insert(fs.kind)
        end
        if not fm[fs.name] then
            fm[fs.name] = self:CompileFunctionSpec(fs)
            if fm.derivedfrom and fs.derivedfrom then
                if fm.derivedfrom ~= fs.derivedfrom then
                    print("not same energy spec?")
                    print(fm.derivedfrom)
                    print(fs.derivedfrom)
                end
            end
            fm.derivedfrom = fm.derivedfrom or fs.derivedfrom
        else
            -- TODO: more principled handling of exclude
            assert(fs.name == "exclude", "function already defined "..fs.name)
        end
    end
    for _,k in ipairs(kinds) do
        local fm = kind_to_functionmap[k]
        self.P:Functions(k,fm)
    end
end

local function createzerolist(N)
    local r = terralib.newlist()
    for i = 1,N do
        r[i] = ad.toexp(0)
    end
    return r
end
    
local function lprintf(ident,fmt,...)
    if true then return end 
    local str = fmt:format(...)
    ident = (" "):rep(ident*4)
    str = ident..str:gsub('\n', "\n"..ident)
    return print(str) 
end

local EMPTY = List()

local function logScatters(fnname,mapname,scattermap)
    dprint(fnname.."scatters, "..mapname..":")
    for k,v in pairs(scattermap) do
        dprint(k)
        dprint(tostring(v.image).."("..tostring(v.index)..")("..tostring(v.channel)..")")
        dprint(v.expression)
    end
end


-------------------------------- JtJp -------------

local UArg = A.UnknownArg
local RArg = A.ResidualArg
local UPArg = A.UnknownPairArg
local function createapplyjtjResidualwise(PS,ES,fnkind)
    local P,Ap_X = PS:UnknownArgument(1),PS:UnknownArgument(2)

    local result = ad.toexp(0)
    local scatters = List() 
    local scattermap = {}
    local function addscatter(u,exp)
        local s = scattermap[u]
        if not s then
            s =  A.Scatter(Ap_X[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local Jp = ad.toexp(0)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            Jp = Jp + partial*P[u.image.name](u.index,u.channel)
        end
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local jtjp = 1.0 * Jp*partial
            result = result + P[u.image.name](u.index,u.channel)*jtjp
            addscatter(u,jtjp)
        end
    end
    logScatters("JTJ", "Ap_X", scattermap)
    return A.FunctionSpec(fnkind,"applyJTJ", List {UArg("P"), UArg("Ap_X")}, List { result }, scatters, ES)
end


--given that the residual at (0,0) uses the variables in 'unknownsupport',
--what is the set of residuals will use variable X(0,0).
--this amounts to taking each variable in unknown support and asking which residual is it
--that makes that variable X(0,0)
local function residualsincludingX00(unknownsupport,unknown,channel)
    assert(channel)
    local r = terralib.newlist()
    for i,u in ipairs(unknownsupport) do
        if u.image == unknown and u.channel == channel then
            r:insert(u.index:Invert())
        end
    end
    return r
end
local function unknownsforresidual(r,unknownsupport)
    return unknownsupport:map(function(u) return u:shift(r) end)
end

local function shiftexp(exp,o)
    local function rename(a)
        return ad.v[a:shift(o)]
    end
    return exp:rename(rename)
end 

local function simplifyscatters(scatters)
    for _,s in ipairs(scatters) do
        s.expression = ad.polysimplify(s.expression)
    end
end

local function createjtjcentered(PS,ES,fnkind)
    local UnknownType = PS.P:UnknownType()
    local zero_off = fnkind.domain:ZeroOffsetIndex()
    local ispace = fnkind.domain:IndexSpace()
    local N = UnknownType:VectorSizeForIndexSpace(ispace)
    local P = PS:UnknownArgument(1)
    local CtC = PS:UnknownArgument(2)
    --local Pre = PS:UnknownArgument(3)
    local P_hat_c = {}
    local conditions = terralib.newlist()
    --print(tostring(#ES.residuals).." ES.residuals")
    for rn,residual in ipairs(ES.residuals) do
        local F,unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"\n\n##################################################")
        lprintf(0,"r%d = %s",rn,F)
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do 
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(zero_off,chan)
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)
            --print(tostring(#residuals).." residuals")
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r)
                local condition,drdx00 = ad.splitcondition(rexp:d(x))
                lprintf(1,"instance:\ndr%d_%s/dx00[%d] = %s",rn,tostring(r),chan,tostring(drdx00))
                lprintf(1,"condition:\n%s",tostring(condition))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local uv = ad.v[u]
                    local condition2, drdx_u = ad.splitcondition(rexp:d(uv))
                    local exp = drdx00*drdx_u
                    lprintf(2,"term:\ndr%d_%s/dx%s[%d] = %s",rn,tostring(r),tostring(u.index),u.channel,tostring(drdx_u))
                    lprintf(2,"condition:\n%s",tostring(condition2))
                    local conditionmerged = condition*condition2
                    if not P_hat_c[conditionmerged] then
                        conditions:insert(conditionmerged)
                        P_hat_c[conditionmerged] = createzerolist(N)
                    end
                    P_hat_c[conditionmerged][idx+1] = P_hat_c[conditionmerged][idx+1] + P[u.image.name](u.index,u.channel)*exp
                end
            end
        end
    end
    local P_hat = createzerolist(N)
    for _,c in ipairs(conditions) do
        for i = 1,N do
            P_hat[i] = P_hat[i] + c*P_hat_c[c][i]
        end
    end
    for i,p in ipairs(P_hat) do
        P_hat[i] = 1.0 * p
    end
    if PS:UsesLambda() then
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local u = unknown(zero_off,chan)
            P_hat[idx+1] = P_hat[idx+1] + CtC[unknownname](zero_off,chan)*P[unknownname](zero_off,chan)
        end
    end
    --print("JTJ[nopoly] = ", ad.tostrings(P_hat))
    P_hat = ad.polysimplify(P_hat)
    --print("JTJ[poly] = ", ad.tostrings(P_hat))
    local r = ad.Vector(unpack(P_hat))
    local result = A.FunctionSpec(fnkind, "applyJTJUnknownwise", List {UArg("P"), UArg("CtC")}, List{r}, EMPTY,ES)
    return result
end

local function createjtfcentered(PS,ES,fnkind)
   local UnknownType = PS.P:UnknownType()
   local ispace = fnkind.domain:IndexSpace()
   local N = UnknownType:VectorSizeForIndexSpace(ispace)
   local zero_off = fnkind.domain:ZeroOffsetIndex()
   local F_hat = createzerolist(N) --gradient
   local P_hat = createzerolist(N) --preconditioner
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))

        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(zero_off,chan)
            
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)

            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)     -- entry of J^T
                local dfdx00F = dfdx00*F_x  -- entry of \gradF == J^TF
                F_hat[idx+1] = F_hat[idx+1] + dfdx00F           -- summing it up to get \gradF

                local dfdx00Sq = dfdx00*dfdx00  -- entry of Diag(J^TJ)
                P_hat[idx+1] = P_hat[idx+1] + dfdx00Sq          -- summing the pre-conditioner up
                lprintf(2,"dR[%d]_%s/dx[%d] = %s",ridx,tostring(f),chan,tostring(dfdx00F))
            end

        end
    end
    for i = 1,N do
        if not PS.P.usepreconditioner then
            P_hat[i] = ad.toexp(1.0)
        else
            P_hat[i] = ad.polysimplify(P_hat[i])
        end
        F_hat[i] = ad.polysimplify(1.0 * F_hat[i])
    end
    dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return A.FunctionSpec(fnkind,"evalJTFUnknownwise", EMPTY, List{ ad.Vector(unpack(F_hat)), ad.Vector(unpack(P_hat)) }, EMPTY,ES)
end

-- Optimizations todo: only compute triangle, fill in rest
local function creatematerializejtjResidualwise(PS,ES,fnkind)
    local JTJ = PS:JTJArgument()
    local result = ad.toexp(0)
    local scatters = List() 
    local scattermap = {}
    local function addscatter(u0,u1,exp)
        local s = scattermap[u0]
        if not s then
            scattermap[u0] = {}
        end
        s = scattermap[u0][u1]
        if not s then
            local image = JTJ[u0.image.name][u1.image.name]
            local index = A.UnknownPairIndex(u0.index,u1.index)
            local channel = u0.channel*u1.image.type.channelcount+u1.channel
            s =  A.Scatter(image,index,channel,ad.toexp(0),"add")
            scattermap[u0][u1] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local Jp = ad.toexp(0)
        for j,partial0 in ipairs(partials) do
            local u0 = unknownsupport[j]
            for k,partial1 in ipairs(partials) do
                local u1 = unknownsupport[k]
                local tmp = partial0*partial1
                addscatter(u0,u1,tmp)
            end
        end
    end
    simplifyscatters(scatters)
    return A.FunctionSpec(fnkind,"materializeJTJ", List {UPArg("JTJ")}, EMPTY, scatters, ES)
end

local function createapplyjResidualwise(PS,ES,fnkind)
    local P,Jp = PS:UnknownArgument(1),PS:ResidualArgument(2)
    local result = ad.toexp(0)
    local scatters = List() 
    local scattermap = {}
    local function addscatter(channel,exp)
        local s = scattermap[channel]
        if not s then
            s =  A.Scatter(Jp[fnkind.name],fnkind.residualindex,channel,ad.toexp(0),"add")
            scattermap[channel] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local jp = ad.toexp(0)
        --print("F = "..tostring(F))
        --print("unknowns = "..tostring(unknownsupport))
        --[[for _,u in ipairs(unknownsupport) do
            print(tostring(u))
        end--]]
        --print("Num partials = "..tostring(#partials))
        for j,partial in ipairs(partials) do
            --print(partial)
            local u = unknownsupport[j]
            jp = jp + partial*P[u.image.name](u.index,u.channel)
        end
        jp = addscatter(i-1,jp)
    end
    simplifyscatters(scatters)
    logScatters("applyJ", "Jp", scattermap)

    return A.FunctionSpec(fnkind,"applyJ", List { UArg("P"), RArg("Jp")}, EMPTY, scatters, ES)
end

local function createcomputejResidualwise(PS,ES,fnkind)
    local result = ad.toexp(0)
    local outputs = List() 
    local scattermap = {}
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for j,partial in ipairs(partials) do
            outputs:insert(partial)
        end
    end
    return A.FunctionSpec(fnkind,"computeJ", EMPTY, outputs, EMPTY, ES)
end


local function createapplyjtResidualwise(PS,ES,fnkind)
    local Jp,Ap_X,P = PS:ResidualArgument(3),PS:UnknownArgument(1),PS:UnknownArgument(2)
    local result = ad.toexp(0)
    local scatters = List()
    local scattermap = {}
    local function addscatter(u,exp)
        local s = scattermap[u]
        if not s then
            s =  A.Scatter(Ap_X[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local jp = Jp[fnkind.name](fnkind.residualindex)
        if #ES.residuals > 1 then
            jp = jp(i-1)
        end
        for j,partial in ipairs(partials) do
            local u = unknownsupport[j]
            local jtjp = 1.0 * jp*partial
            result = result + P[u.image.name](u.index,u.channel)*jtjp
            addscatter(u,jtjp)
        end
    end
    simplifyscatters(scatters)
    logScatters("applyJt", "Ap_X", scattermap)

    return A.FunctionSpec(fnkind,"applyJt", List { UArg("Ap_X"), UArg("P"), RArg("Jp") }, List { result }, scatters, ES)
end

-------------------------------- JtJp (END) -------------

local function createmodelcostResidualwise(PS,ES,fnkind)
    local Delta = PS:UnknownArgument(1)
    local result = 0.0 --model residuals squared (and summed among unknowns...)
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)

        local JTdelta = 0.0

        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local delta = Delta[u.image.name](u.index,u.channel)
            JTdelta = JTdelta + (partial * delta)
        end
        local residual_m = F + JTdelta
        result = result + (residual_m*residual_m)
    end
    result = ad.polysimplify(ad.toexp(0.5)*result)
    return A.FunctionSpec(fnkind, "modelcost", List { UArg("Delta") }, List{ result }, EMPTY,ES)
end

local function createjtfResidualwise(PS,ES,fnkind)
    local R,Pre = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List()
    local scattermap = { [R] = {}, [Pre] = {}}
    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            --print(u)
            s =  A.Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    --print("createJTF")
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        --print("unknowns ")
        --[[
        unknownsupport:map(function(x) 
            print(" "..tostring(x))
            return nil 
        end)
        --]]
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        --[[print("unknown vars ")
        unknownvars:map(function(x) 
            print(" "..tostring(x))
            return nil 
        end)--]] 
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            addscatter(R,u,-1.0*partial*F)
            addscatter(Pre,u,partial*partial)
        end
    end
    simplifyscatters(scatters)
    logScatters("JTF", "R", scattermap[R])
    logScatters("JTF", "Pre", scattermap[Pre])
    return A.FunctionSpec(fnkind, "evalJTF", List { UArg("R"), UArg("Pre") }, EMPTY, scatters, ES)
end


local function computeCtCResidualwise(PS,ES,fnkind)
    local CtC = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List() 
    local scattermap = { [CtC] = {}}

    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            s =  A.Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local inv_radius = 1.0 / PS.trust_region_radius
            addscatter(CtC,u,partial*partial*inv_radius)
        end
    end
    simplifyscatters(scatters)
    return A.FunctionSpec(fnkind, "computeCtC", List { UArg("CtC") }, EMPTY, scatters, ES)
end
    
local function createcost(ES, fnkind)
    local function sumsquared(terms)
        local sum = ad.toexp(0)
        for i,t in ipairs(terms) do
            sum = sum + t*t
        end
        return ad.toexp(0.5)*sum
    end
    local exp = sumsquared(ES.residuals:map("expression"))
    return A.FunctionSpec(fnkind,"cost", EMPTY, List{exp}, EMPTY,ES) 
end

-- Return domains ordered with the domains specified by domain_order first
-- (in order), followed by any domains not in the ordering
local function set_domain_order(domains, domain_order)
    local result = List()
    local used = {}
    for _,d in ipairs(domain_order) do
        if table.indexOf(domains, d) >= 0 then
            used[d] = true
            result:insert(d)
        end
    end
    for _,d in ipairs(domains) do
        if not used[d] then
            result:insert(d)
        end
    end
    return result
end

local function getclassifyexpression(domain_order)
    return function (exp) -- what this thing is mapped over
        local domains = {}
        local externaldomains = {}
        local function adddomainlist(domainlist, internal_domains)
            for _,d in ipairs(domainlist) do 
                domains[d] = d
                if table.indexOf(internal_domains, d) < 0 then
                    externaldomains[d] = d 
                end
            end 
        end
        local function visit_fn(a,internal_domains)
            if A.ImageAccess:isclassof(a) then
                local newdomains = util.extract_domains_from_index(a.index)
                adddomainlist(newdomains, internal_domains)
                if a.image.tensor_contraction then
                    local new_internal_domains = List()
                    new_internal_domains:insertall(internal_domains)
                    new_internal_domains:insertall(a.image.contracted_domains)
                    a.image.expression:visit(visit_fn,new_internal_domains) -- Recurse
                end
            elseif A.IndexValue:isclassof(a) then
                local newdomains = List({a.indexdomain})
                adddomainlist(newdomains, internal_domains)
            elseif A.SparseIndexValue:isclassof(a) then
                local newdomains = util.extract_domains_from_index(a.access.index)
                adddomainlist(newdomains, internal_domains)
            end
        end
        exp:visit(visit_fn, List())
        local unknownaccesses = extract_unknowns(exp)
        local template = A.ResidualTemplate(exp,unknownaccesses)
        if next(domains) == nil then
            error("residual must actually use some image")
        end
        --[[ TODO: handle out of bounds again.
        if classification.kind == "CenteredFunction" then
            exp:visit(function(a)
                if BoundsAccess:isclassof(a) and #a.min.data ~= #classification.ispace.dims then
                    error(string.format("%s does not match index space %s",a,classification.ispace.dims))
                end
            end)
            -- by default zero-out any residual computation that uses out-of-bounds things
            -- users can opt for per-residual custom behavior using the InBounds checks
        end
        --]]

        local function sortdomains(ds)
            -- Deterministic sorting
            local unsorteddomains = {}
            local sorteddomains = terralib.newlist()
            for _,v in pairs(ds) do
                unsorteddomains[#unsorteddomains+1] = v
            end
            return sort_index_domains(unsorteddomains)
        end

        local sorteddomains           = sortdomains(domains)
        local sortedexternaldomains   = sortdomains(externaldomains)
        if domain_order then
            sorteddomains         = set_domain_order(sorteddomains, domain_order)
            sortedexternaldomains = set_domain_order(sortedexternaldomains, domain_order)
        end

        local iterationDomain = A.IterationDomain(sorteddomains)
        local externaliterationDomain = A.IterationDomain(sortedexternaldomains)
        local constraints = constraintsforexpression(iterationDomain,exp)
        template.expression = ad.select(constraints:asvar(),exp,ad.toexp(0.0))
        local resDom = A.ResidualDomain(iterationDomain,externaliterationDomain)
        --print("resDom")
        --print(resDom)
        return resDom,template
    end
end

function createprecomputed(precomputeddomains)
    local itdoms,pd_map = MapAndGroupBy(precomputeddomains,function(pd) return pd.domain,pd end)
    local precomputes = List()
    for _,itdom in ipairs(itdoms) do
        local scatters = List()
        local scatterindex = A.ImageIndex(itdom.domains:map(function(dom)
                return A.DirectIndexComponent(dom)
            end))
        for _,pd in ipairs(pd_map[itdom]) do
            local im = pd.im
            local expression = ad.polysimplify(im.expression)
            local scattermode = "set"
            if pd.resultdomain then
                scattermode = "add"
                scatterindex = A.ImageIndex(pd.resultdomain.domains:map(function(dom)
                    return A.DirectIndexComponent(dom)
                end))

                --Insert a dummy clear function
                local clearscatters = List()
                clearscatters:insert(A.Scatter(im, scatterindex, 0, ad.toexp(0), "set"))
                local fnkind = A.IterationDomainwiseFunction(pd.resultdomain)
                local pc = A.FunctionSpec(fnkind,"precompute", EMPTY, EMPTY, clearscatters)
                precomputes:insert(pc)
                --TODO: gradients
                print("TODO: Need clear of precompute gradient arrays")
            end
            if im:Materialized() then
                scatters:insert(A.Scatter(im, scatterindex, 0, expression, scattermode))
            end
            local gim = im.gradientimage
            if gim then
                for i,expression in ipairs(gim.image.expressions) do
                    local gradientexpression = ad.polysimplify(expression)
                    if gim.image:Materialized(i) then
                        scatters:insert(A.Scatter(gim.image, scatterindex, i-1, gradientexpression, scattermode))
                    end
                end
            end
        end
        if #scatters > 0 then
            local fnkind = A.IterationDomainwiseFunction(itdom)
            local pc = A.FunctionSpec(fnkind,"precompute", EMPTY, EMPTY, scatters)
            precomputes:insert(pc)
        end
        
    end
    return precomputes
end

local function DefaultSchedule()
    return A.Schedule(A.INLINE, A.ResidualSchedule, A.ResidualSchedule)
end

local function get_schedule(named_residual)
    local store_jtj = named_residual.JtJ.materialize
    local store_jp  = named_residual.Jp.materialize
    local store_j   = named_residual.J.materialize
    local compute_at_output = named_residual._compute_at_output

    local jtf_compute_at_output = named_residual.JtF._compute_at_output

    local fnschedule = compute_at_output and 
            A.AtOutputSchedule or
            A.ResidualSchedule
    local jtfschedule = jtf_compute_at_output and 
            A.AtOutputSchedule or
            A.ResidualSchedule
    local jtjp_schedule
    if not (store_jtj or store_jp or store_j) then
        jtjp_schedule = A.INLINE
    elseif (store_jtj and (not store_j)) then
        assert(not store_jp, "Cannot simultaneously materialize JtJ and Jp")
        jtjp_schedule = A.PRECOMPUTE_JTJ(named_residual.JtJ.sparse)
    elseif (store_jp and (not store_j))then
        jtjp_schedule = A.APPLY_SEPARATELY
    elseif store_jtj and store_j then
        assert(not store_jp, "Cannot simultaneously materialize JtJ and Jp")
        jtjp_schedule = A.PRECOMPUTE_J_THEN_JTJ(named_residual.JtJ.sparse)
    elseif store_j then
        if not store_jp then
            print("If materializing J but not JTJ, then must also materialize Jp, enabling Jp materialization.")
        end
        jtjp_schedule = A.PRECOMPUTE_J(named_residual.J.sparse)
    else
        assert(false, "Invalid JtJp schedule, must choose from:\n\tJtJp\n\t[JtJ]p\n\tJt[Jp]\n\t[Jt][[J]p]\n\t[[Jt][J]]p\n")
    end
    return A.Schedule(jtjp_schedule, jtfschedule, fnschedule)
end

local function get_unknown_iteration_domain(unknowns)
    local function get_it_dom(imacc)
        return A.IterationDomain(util.extract_domains_from_index(imacc.index))
    end
    local itdom = get_it_dom(unknowns[1])
    for i=2,#unknowns do
        if (itdom ~= get_it_dom(unknowns[i])) then
            return nil
        end
    end
    return itdom
end

local function toresidualgroups(energy)
    local domainschedToTemplates = {}
    local das_order = List()
    for _,residual in ipairs(energy.residuals) do
        local name = residual.name
        local Rs = residual.expressions
        local schedule = get_schedule(residual)
        for i,exp in ipairs(Rs) do 
            local usestc = false
            exp:visit(function(a)
                    if A.ImageAccess:isclassof(a) and a.image.tensor_contraction then
                        usestc = true
                        return
                    end
                end)
            if usestc then
                --print("Uses tensor_contraction")
                --print(schedule)

                local validTensorContractionJtJpSchedules = {A.APPLY_SEPARATELY, A.PRECOMPUTE_J, A.PRECOMPUTE_J_THEN_JTJ}
                assert( util.exists(validTensorContractionJtJpSchedules, function(t) return t:isclassof(schedule.jtjpschedule) end), 
                    "Using a tensor contraction for residual, must use Jt[Jp], [[Jt][J]]p, or [Jt][[J]p] schedule")
            end
        end
        local domains,domains_to_templates = MapAndGroupBy(Rs,getclassifyexpression(residual:domain_order()))
        for _,domain in ipairs(domains) do
            local das = A.DomainAndSchedule(domain,schedule)
            if domainschedToTemplates[das] then 
                print("Merging residuals with identical schedules")
            else
                das_order:insert(das)
                domainschedToTemplates[das] = List()
            end
            for _,v in ipairs(domains_to_templates[domain]) do
                domainschedToTemplates[das]:insert(v)
            end
        end
    end 
    local residual_groups = List()
    for _,das in ipairs(das_order) do
        residual_groups:insert(A.ResidualGroup(das,domainschedToTemplates[das]))
    end
    for _,rg in ipairs(residual_groups) do
        local schedule = rg.domainandschedule.schedule
        if A.AtOutputSchedule:isclassof(schedule.fnschedule) then
            for _,t in ipairs(rg.residuals) do
                assert(rg.domainandschedule.domain.full == get_unknown_iteration_domain(t.unknowns))
            end
        end
    end
    print("Residual Group Count: "..tostring(#residual_groups))
    return residual_groups
end

local direct_solve_threshold = 64
local dense_materialize_threshold = 256
local direct_solve_unknown_count_threshold = 0

local function has_reduction(named_residual)
    local usestc = false
    local has_tc_fun = function(a)
        if A.ImageAccess:isclassof(a) and a.image.tensor_contraction then
            usestc = true
            return
        end
    end
    for i,exp in ipairs(named_residual.expressions) do
        exp:visit(has_tc_fun)
    end
    return usestc
end

local W = io.open("schedules.txt","a")
W:write("\n")
W:close()

local function logSchedule(schedule)
    local W = io.open("schedules.txt","a")
    W:write(tostring(schedule))
    W:close()
end

local function writeScheduleResults_lua(cost, time, index, estimated_cost)
    local W = io.open("schedules.txt","a")
    W:write("Cost: ")
    W:write(cost)
    W:write("\n")
    W:write("Total: ")
    W:write(time)
    W:write("ms \n")
    W:write("Index: ")
    W:write(index)
    W:write("\n")
    W:write("Analysis ")
    W:write(index)
    W:write(" ")
    W:write(estimated_cost)
    W:write(" ")
    W:write(time)
    W:write("\n")
    W:close()
end
writeScheduleResults = terralib.cast({double,double,int, double} -> {}, writeScheduleResults_lua)

local function normalize_energy(energy)
    -- Split residuals fully
    local res_count = #energy.residuals
    repeat
        res_count = #energy.residuals
        for _,r in ipairs(energy.residuals) do
            energy:full_split(r)
        end
    until res_count == #energy.residuals
end


local function cost_model(explike)
    local compute_cost = 0
    local memory_cost = 0
end

local function default_residual_group_schedule(expressions)
    local domain_order = get_index_domains_from_exp_list(expressions)
    return A.ResidualGroupSchedule(false, false, false, false, false, domain_order, List())
end

local function copy_residual_schedule(base_schedule)
    local b = base_schedule
    return A.ResidualGroupSchedule(b.jtj_materialize,
        b.j_materialize,
        b.jp_materialize, 
        b.compute_at_output, 
        b.jtf_compute_at_output,
        b.domain_order,
        b.sumstoparallelize)
end

local function default_schedule(energy)
    local rgschedules = List()
    local names = List()
    for i,rg in ipairs(energy.residuals) do
        names:insert(rg.name)
        rgschedules:insert(default_residual_group_schedule(rg.expressions))
    end
    return A.FullSchedule(rgschedules, List(), List(), List())
end

local function can_compute_jtf_at_output(rg,domain_order)
    local can = true
    local itdom = A.IterationDomain(domain_order)
    local classifyexp = getclassifyexpression()
    for _,exp in ipairs(rg.expressions) do
        local _,t = classifyexp(exp)
        local unknown_domains = get_unknown_iteration_domain(t.unknowns)
        if not unknown_domains then 
            return false
        end
        can = can and (itdom == unknown_domains)
        for _,u in ipairs(t.unknowns) do
            for _,c in ipairs(u.index.components) do
                if A.SparseIndexComponent:isclassof(c) then
                    return false
                end
            end
        end
    end
    return can
end

local function generate_all_residual_schedules(rg,include_JtJ_materialize,include_J_materialize)
    local all_schedules = List()
    local materializej = {false}
    local materializejp = {true}
    local materializejtj = {false}

    local function add_to_materialize_lists(j,jp,jtj)
        materializej[#materializej+1] = j
        materializejp[#materializejp+1] = jp
        materializejtj[#materializejtj+1] = jtj
    end
    if not has_reduction(rg) then
        add_to_materialize_lists(false,false,false)
    end


    if include_JtJ_materialize and not has_reduction(rg) then
        add_to_materialize_lists(false,false,true)
    end

    if include_J_materialize then
        add_to_materialize_lists(true,false,false)
        if include_JtJ_materialize then
            add_to_materialize_lists(true,false,true)
        end
    end

    local original_domain_order = rg.domains
    for i=1,#materializej do
        for domain_order in util.permutations(original_domain_order) do
            domain_order = List({unpack(domain_order)}) -- Make a copy
            local compute_at_outputs = {false}
            local jtf_compute_at_outputs = {false}
            if can_compute_jtf_at_output(rg,domain_order) then
                if materializejp[i] then
                    --print("TODO: remove restriction on materializejp and jtf_compute")
                else
                    jtf_compute_at_outputs[2] = true
                end
            end
            if jtf_compute_at_outputs[2] and not (materializejtj[i] or materializej[i] or materializejp[i]) then
                compute_at_outputs[2] = true
            end
            for _,compute_at_output in ipairs(compute_at_outputs) do
                for _,jtf_compute_at_output in ipairs(jtf_compute_at_outputs) do
                    local new_schedule = A.ResidualGroupSchedule(materializejtj[i],
                        materializej[i],
                        materializejp[i], 
                        compute_at_output, 
                        jtf_compute_at_output,
                        domain_order,
                        List())
                    all_schedules:insert(new_schedule)
                end
            end
        end
    end
    return all_schedules
end

local function same_rg_schedule(rg0, rg1)
    local same_domains = true
    for i=1,#rg0.domain_order do
        same_domains = same_domains and (rg0.domain_order[i] == rg1.domain_order[i])
    end
    local same = same_domains and (rg0.jtj_materialize == rg1.jtj_materialize)
        and (rg0.j_materialize == rg1.j_materialize)
        and (rg0.jp_materialize == rg1.jp_materialize)
        and (rg0.compute_at_output == rg1.compute_at_output)
        and (rg0.jtf_compute_at_output == rg1.jtf_compute_at_output)
    return same 
end



local function generate_all_schedules(energy, unknown_count, count_only)
    local all_schedules = List()
    local dense_JtJ_size = unknown_count*unknown_count
    local max_memory = 0.7*util.global_memory
    print("WARNING: Setting max memory to that of lothlann for rapid dev; this artificially expands the search space on low-memory machines")

    max_memory = 8950985523.2 -- TODO: remove; this is just to pretend to be lothlann for rapid dev

    local element_size = 4
    if thallo_float == double then
        element_size = 8
    end
    local include_JtJ_materialize = dense_JtJ_size*element_size < max_memory
    local sparse_J_size = 0

    do
        --Default Schedule
        for _,rg in ipairs(energy.residuals) do
            rg.JtJ:set_materialize(false)
            rg.Jp:set_materialize(true)
            rg.J:set_materialize(false)
        end
        local rgs = toresidualgroups(energy)
        for _,rg in ipairs(rgs) do
            sparse_J_size = sparse_J_size + rg:jacobianEntriesPerElement()*rg:cardinality()
        end
    end

    local include_J_materialize = (sparse_J_size*(element_size + 8)*20) < max_memory

    local residual_schedules = List()
    local names = List()
    for _,r in ipairs(energy.residuals) do
        names:insert(r.name)
        residual_schedules:insert(generate_all_residual_schedules(r,include_JtJ_materialize,include_J_materialize))
    end
    local i = 0
    for residual_schedule in util.cartesian_product(residual_schedules) do
        local compute_at_output_count = 0
        local compute_at_output_jtf_count = 0
        local materialize_j_only_count = 0
        local materialize_j_then_jtj_count = 0
        local materialize_jtj_only_count = 0
        local all_same = true
        local invalid_j_combo = false
        for _,sched in ipairs(residual_schedule) do
            if sched.jtf_compute_at_output then compute_at_output_jtf_count = compute_at_output_jtf_count + 1 end
            if sched.compute_at_output then compute_at_output_count = compute_at_output_count + 1 end
            if sched.j_materialize or sched.jtj_materialize then
                if sched.j_materialize and sched.jtj_materialize then
                    materialize_j_then_jtj_count = materialize_j_then_jtj_count + 1
                    invalid_j_combo = invalid_j_combo or ((materialize_j_only_count+materialize_jtj_only_count) > 0)
                elseif sched.j_materialize  then
                    materialize_j_only_count = materialize_j_only_count + 1
                    invalid_j_combo = invalid_j_combo or ((materialize_j_then_jtj_count+materialize_jtj_only_count) > 0)
                else --sched.jtj_materialize
                    materialize_jtj_only_count = materialize_jtj_only_count + 1
                    invalid_j_combo = invalid_j_combo or ((materialize_j_only_count+materialize_j_then_jtj_count) > 0)
                end
            end
            all_same = all_same and same_rg_schedule(sched,residual_schedule[1])
        end
        if invalid_j_combo or ((not all_same) and (compute_at_output_count > 1 or compute_at_output_jtf_count > 1)) then
            -- print("TODO: Remove restriction on only one compute_at_output if not identical")
            -- print("TODO: Remove restriction on combining materialize J and J-then-JtJ")
        else
            local materialized_expressions = List()
            for k,_ in pairs(ComputedArrayCache) do
                materialized_expressions:insert(k)
            end
            materialized_expressions:sort(function(e1,e2)
                return ComputedArrayCache[e1].computed_array_index < ComputedArrayCache[e2].computed_array_index
            end)
            local ps_mat_exp = util.powerset(materialized_expressions)

            for _,inlined_exps in ipairs(ps_mat_exp) do
                for _,inlined_grad_exps in ipairs(ps_mat_exp) do
                    if not count_only then
                        all_schedules:insert(A.FullSchedule(List(residual_schedule), names, inlined_exps, inlined_grad_exps))
                        if (i+1) % 100000 == 0 then print("Generated "..tostring(i+1).." schedules") end
                    end
                    i = i + 1
                end
            end
        end
    end
    print("Total Schedule Count: "..tostring(i))
    --assert(false)
    return all_schedules,i
end

local function apply_schedule(energy,schedule)
    assert(#schedule.rgschedules == #energy.residuals)
    for i,rg in ipairs(energy.residuals) do
        local rs = schedule.rgschedules[i]
        rg.JtJ:set_materialize(rs.jtj_materialize)
        rg.Jp:set_materialize(rs.jp_materialize)
        rg.J:set_materialize(rs.j_materialize)
        rg:compute_at_output(rs.compute_at_output)
        rg.JtF:compute_at_output(rs.jtf_compute_at_output)
        rg:reorder(rs.domain_order)
    end

    for k,_ in pairs(ComputedArrayCache) do
        k:set_materialize(true)
    end

    for _,exp in ipairs(schedule.exptoinline) do
        exp:set_materialize(false)
    end
    for _,exp in ipairs(schedule.exptoinlinegradient) do
        exp:set_materialize(false)
    end
end


local function cost_of_kernel(k)

    -- KernelCostData = (number threadcount, number opcount_per_thread, number memreads_per_thread, number memwrites_per_thread)
    -- $$C(k) = \frac{t_k}{T(r_k)} \max(m_k/t_m, o_k/t_c),$$
    -- where $o_k$ is the number of arithmetic operations in kernel $k$, 
    -- $t_k$ is the number of threads launched by kernel $k$, 
    -- $m_k$ is the amount of memory accessed in the kernel, where atomic writes are only counted once per warp if the iteration order (specified by \ic{reorder()}) allows for warp reductions before the atomic write.
    
    --[[
    For this cost model, we need to query a few properties of the GPU: the theoretical peak FLOPs ($t_c$), 
    the peak bandwidth ($t_m$), the maximum available memory ($M$), the number of threads in a warp $W$, 
    and a function that takes the estimated number of registers used by a kernel ($r_k$) and returns 
    the number of concurrent threads that can be active at once ($T(r)$).
    --]]
    local theoretical_peak_flops = util.cu.theoretical_peak_flops
    local peak_bandwidth_bytes_per_sec = util.cu.theoretical_memory_bandwidth
    local max_gpu_memory = util.cu.global_memory
    local max_concurrent_threads = util.cu.get_max_active_threads(k.register_count) -- T(r_k); TODO: memoize?

    local scalar_size = 4

    --$$C(k) = \frac{t_k}{T(r_k)} \max(m_k/t_m, o_k/t_c),$$
    -- TODO: Consider changing this to # of waves (ceiling function of this value)
    --local thread_multiplier = k.thread_count / max_concurrent_threads -- t_k/T(r_k)

    -- Concurrency multiplier (test)
    local thread_multiplier = 1.0
    local register_multiplier = util.cu.get_max_active_threads(15)/util.cu.get_max_active_threads(k.register_count)
    --local thread_multiplier = math.min(max_concurrent_threads/k.thread_count,1.0)
    --local thread_multiplier = math.ceil(k.thread_count / max_concurrent_threads)

    local memory_term = (k.memreads_per_thread+k.memwrites_per_thread)*k.thread_count*scalar_size / peak_bandwidth_bytes_per_sec -- m_k/t_m
    local compute_term = (k.opcount_per_thread)*k.thread_count / theoretical_peak_flops --o_k/t_c

    --print("Memory/Compute Ratio: "..tostring(memory_term/compute_term))

    return thread_multiplier * register_multiplier * math.max(memory_term, compute_term) 
end

local function cost_of_scheduled_energy(psad,scheduledenergy,lin_iter_hint,verbose)
    local use_dense_where_possible = true
    local compute_intermediate_cost = false
    psad:SetScheduledEnergy(scheduledenergy)
    local cost = 0

    local uses_lambda = psad.P:UsesLambda()
    local unknown_cardinality = psad.P:UnknownType():cardinality()

    local use_cublas = scheduledenergy:RequiresJtJMaterialize()--use_direct_solve
    local use_cusparse = scheduledenergy:RequiresJ()

    -- TODO: compute multiple kernels (one per ispace?)
    local unknownwise_clear_kernel = A.KernelCostData(unknown_cardinality, 8, 0, 0, 1)
    local unknownwise_copy_kernel = A.KernelCostData(unknown_cardinality, 8, 0, 1, 1)
    local nnz = 0
    local residual_cardinality = 0
    local materialized_is_dense = true
    local materializeJScheds = {A.PRECOMPUTE_J, A.PRECOMPUTE_J_THEN_JTJ}
    local nnzMaterialized = 0
    local nResidualsMaterialized = 0
    local function numberofelements(RG)
        return RG:cardinality()
    end
    
    local total_nentries = 0
    local total_possible_nentries = 0
    local all_unknowns = {}
    local unknown_count = 0
    for _,rg in ipairs(scheduledenergy.residualgroups) do
        local residuals_per_element = #rg.residuals            
        local nentries = rg:jacobianEntriesPerElement()
        residual_cardinality = residual_cardinality + numberofelements(rg)*residuals_per_element
        nnz = nnz + numberofelements(rg)*nentries
        local jtjp_sched = rg.domainandschedule.schedule.jtjpschedule
        if util.exists(materializeJScheds, function(t) return t:isclassof(jtjp_sched) end) then
            nResidualsMaterialized = nResidualsMaterialized + numberofelements(rg)*residuals_per_element
            nnzMaterialized = nnzMaterialized + numberofelements(rg)*nentries
            total_nentries = total_nentries + nentries
            total_possible_nentries = total_possible_nentries + (residuals_per_element*unknown_cardinality)
        end
        local domain = rg.domainandschedule.domain
        for _,res in ipairs(rg.residuals) do
            --print("TODO: more accurately estimate JtJ nonzeros for cusparse when there are tensor contractions")
            --print("TODO: more accurately estimate JtJ at all...")
            for _,u in ipairs(res.unknowns) do
                -- TODO: check this 
                if not all_unknowns[u] then
                    all_unknowns[u] = true
                    if domain.external ~= domain.full then
                        local contraction_size = domain.full:IndexSpace():cardinality() / domain.external:IndexSpace():cardinality()
                        print("Unknowns under contraction: "..contraction_size)
                        unknown_count = unknown_count + contraction_size
                    else
                        unknown_count = unknown_count + 1
                    end
                end
            end
        end

    end
    local estimatedNnzJtJ = unknown_cardinality*unknown_count
    -- clamp to feasible
    estimatedNnzJtJ = math.min(estimatedNnzJtJ,unknown_cardinality*unknown_cardinality)
    print("Estimated NnZ JtJ: "..tostring(estimatedNnzJtJ))
    local multistep_alphaDenominator_compute = not scheduledenergy:CanFuseJtJpReduction()

    local density = total_nentries / total_possible_nentries
    -- TODO: Make this more principled
    materialized_is_dense = (density == 1.0) and use_dense_where_possible

    -- KernelCostData = (number thread_count, number register_count, number opcount_per_thread, number memreads_per_thread, number memwrites_per_thread)
    -- This is pretty explicitly built by copying the solver code and modifying it. Ideally we could generate this code from a common
    -- IR used to generate the solver code as well.
    local nonlinear_kernels = {}
    local linear_kernels = {}

    local do_clears = not (scheduledenergy:CanFuseJtJpReduction() and scheduledenergy:CanFusePCGInit()) 

     -- not use_direct_solve, TODO: support direct solve in cost evaluation
    if do_clears then
        nonlinear_kernels["delta_clear"] = unknownwise_clear_kernel
        nonlinear_kernels["Ap_X_clear"] = unknownwise_clear_kernel
    end
    
    if do_clears then
        nonlinear_kernels["r_clear"] = unknownwise_clear_kernel
    end

    -- Throughout, we count special operations (sin/cos/sqrt/div) as 4 operations, and basic add/sub/mul as 1. 
    -- This isn't exact, but roughly corresponds to the relative arithmetic throughputs throughout the 
    -- different CUDA generations: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions

    -- TODO: Experimentally Validate
    -- TODO: pass in unknown_type?
    local unknown_type = psad.P:UnknownType()

    local function computeUnknownwiseKernelCost_raw(params, base_registers, base_ops, pixel_count, channelcount)
        local c = channelcount
        local register_count = base_registers + params.pointer_variable_count * 2 + params.scalar_variable_count + (params.unknown_variable_count * c)
        local op_count = base_ops + (params.unknown_ops * c) + params.unknown_to_scalar_ops*(c - 1) + params.scalar_ops + params.scalar_reductions * 5
        local mem_read_count = (params.unknown_mem_reads * c) + params.scalar_mem_reads
        local mem_write_count = (params.unknown_mem_writes * c) + params.scalar_mem_writes + params.scalar_reductions*(1.0/util.cu.warpSize)
        return A.KernelCostData(pixel_count, register_count, op_count, mem_read_count, mem_write_count)
    end

    local function computeUnknownwiseKernelCost(params, base_registers, base_ops, im)
        return computeUnknownwiseKernelCost_raw(params, base_registers, base_ops, im.imagetype.ispace:cardinality(), im.imagetype.channelcount)
    end

    local function computeSimpleResidualWrapper(thread_count, is_reduction, pointer_params)
        local registers = 8 + pointer_params * 2
        local op_count = 8
        local memreads = 0
        local memwrites = 0
        if is_reduction then
            registers = registers + 3
            op_count = op_count + 5
            memreads = (1.0/util.cu.warpSize)
            memwrites = (1.0/util.cu.warpSize)
        end
        return A.KernelCostData(thread_count, registers, op_count, memreads, memwrites)
    end

    local function addUnknownwiseKernels(params)
        assert('number' == type(params.scalar_mem_writes))
        local base_registers = 8
        local base_ops = 8
        for _,im in ipairs(unknown_type.images) do
            local name = params.base_name.."_"..im.name
            params.kernel_table[params.base_name.."_"..im.name] = computeUnknownwiseKernelCost(params, base_registers,base_ops,im) 
        end
    end


    local function mergeCostData(d0, d1)
        return A.KernelCostData(d0.thread_count, 
            d0.register_count + d1.register_count, 
            d0.opcount_per_thread + d1.opcount_per_thread, 
            d0.memreads_per_thread  + d1.memreads_per_thread, 
            d0.memwrites_per_thread + d1.memwrites_per_thread)
    end

    local function addFunctionSpecs(kernel_list, functionspecs)
        for _,fs in ipairs(functionspecs) do
            kernel_list[fs.name.."_"..tostring(fs.kind)] = getfunctioncostdata(psad.P, fs)
        end
    end

    local function addFunctionSpec(kernel_list, name, fs, wrapper)
        kernel_list[name.."_"..tostring(fs.kind)] = mergeCostData(getfunctioncostdata(psad.P, fs), wrapper)
    end


    local usepreconditioner = true -- TODO: Acquire this programmatically

    local guarded_invert_opcount = 10


    local nonlin_functionspecs = List()
    local lin_functionspecs = List()
    for i,residualgroup in ipairs(scheduledenergy.residualgroups) do
        local schedule = residualgroup.domainandschedule.schedule
        local is_unknown_schedule = A.AtOutputSchedule:isclassof(schedule.fnschedule)
        local is_unknown_jtf_schedule = A.AtOutputSchedule:isclassof(schedule.jtfschedule)
        local fnschedule = A.ResidualSchedule
        local domain = residualgroup.domainandschedule.domain
        

        local fnkind = A.ResidualwiseFunction(domain.external,fnschedule)
        fnkind.residualindex = A.ImageIndex(domain.external.domains:map(function(dom)
            return A.DirectIndexComponent(dom)
        end))
        fnkind.name  = "residual_"..tostring(i)
        local full_fnkind = fnkind
        if domain.external ~= domain.full then
            full_fnkind = A.ResidualAndContractionwiseFunction(domain.full,fnschedule)
            full_fnkind.residualindex = A.ImageIndex(domain.external.domains:map(function(dom)
                return A.DirectIndexComponent(dom)
            end))
            full_fnkind.name = fnkind.name
        end

        local residual_cardinality = residualgroup:cardinality()
        local full_cardinality = domain.full:IndexSpace():cardinality()
        if uses_lambda or compute_intermediate_cost then
            local computeCostWrapper = A.KernelCostData(residual_cardinality, 
                                                        9, -- register count 
                                                        8+5, -- op count
                                                        (1.0/util.cu.warpSize), -- mem_read_count
                                                        (1.0/util.cu.warpSize)) -- mem_write_count
            addFunctionSpec(nonlinear_kernels, "computeCost", createcost(residualgroup,fnkind), computeCostWrapper)
        end
        

        local jtjp = schedule.jtjpschedule

        -- PCGInit1
        --jtf
        if is_unknown_jtf_schedule then
            local params = {
                unknown_variable_count=2, scalar_variable_count=1, pointer_variable_count=4,
                unknown_ops=0, unknown_to_scalar_ops=1, scalar_ops=0, 
                unknown_mem_reads=0, scalar_mem_reads=0, 
                unknown_mem_writes=3, scalar_mem_writes=0,
                scalar_reductions=0
            }
            if scheduledenergy:CanFusePCGInit() then
                params.unknown_variable_count = params.unknown_variable_count + 1            
                params.pointer_variable_count = params.pointer_variable_count + 1
                params.unknown_ops = params.unknown_ops + 1
                params.unknown_to_scalar_ops = params.unknown_to_scalar_ops + 1
                params.scalar_ops = params.scalar_ops + 10
                params.unknown_mem_writes = params.unknown_mem_writes + 1
                params.scalar_reductions = 1
            end
            local ispace = fnkind.domain:IndexSpace()
            local pixel_count = ispace:cardinality()
            local channelcount = psad.P:UnknownType():VectorSizeForIndexSpace(ispace)
            assert(channelcount and channelcount > 0)
            local PCGInit1Wrapper = computeUnknownwiseKernelCost_raw(params, 8, 8, pixel_count, channelcount)
            local unknownwise_fnkind = A.UnknownwiseFunction(fnkind.domain)
            local jtfcentered = createjtfcentered(psad,residualgroup,unknownwise_fnkind)
            addFunctionSpec(nonlinear_kernels, "PCGInit1", jtfcentered, PCGInit1Wrapper)
        else
            local PCGInit1Wrapper = computeSimpleResidualWrapper(full_cardinality, false, 2)
            addFunctionSpec(nonlinear_kernels, "PCGInit1", createjtfResidualwise(psad,residualgroup,full_fnkind), PCGInit1Wrapper)
        end

        --jtjp
        if A.INLINE:isclassof(jtjp) then
            -- PCGStep1
            if is_unknown_schedule then
                local params = {
                    unknown_variable_count=1, scalar_variable_count=1, pointer_variable_count=4,
                    unknown_ops=0, unknown_to_scalar_ops=0, scalar_ops=0, 
                    unknown_mem_reads=0, scalar_mem_reads=0, 
                    unknown_mem_writes=1, scalar_mem_writes=0,
                    scalar_reductions=0
                }
                if scheduledenergy:RequiresJtJMaterialize() and scheduledenergy:RequiresJ() then
                    params.unknown_ops = params.unknown_ops + 1
                    params.unknown_mem_reads = params.unknown_mem_reads + 1
                end
                if not multistep_alphaDenominator_compute then
                    params.scalar_variable_count = params.scalar_variable_count+1   
                    params.pointer_variable_count = params.pointer_variable_count + 1
                    params.unknown_ops = params.unknown_ops + 1
                    params.unknown_to_scalar_ops = params.unknown_to_scalar_ops + 1
                    params.scalar_reductions = 1
                end
                local ispace = fnkind.domain:IndexSpace()
                local pixel_count = ispace:cardinality()
                local channelcount = psad.P:UnknownType():VectorSizeForIndexSpace(ispace)
                assert(channelcount and channelcount > 0)
                local PCGStep1Wrapper = computeUnknownwiseKernelCost_raw(params, 8, 8, pixel_count, channelcount)
                local unknownwise_fnkind = A.UnknownwiseFunction(fnkind.domain)
                local jtjcentered = createjtfcentered(psad,residualgroup,unknownwise_fnkind)
                addFunctionSpec(linear_kernels, "PCGStep1", jtjcentered, PCGStep1Wrapper)
            else
                local do_reduce = not multistep_alphaDenominator_compute
                local PCGStep1Wrapper = computeSimpleResidualWrapper(residual_cardinality, do_reduce, 3)
                addFunctionSpec(linear_kernels, "PCGStep1", createapplyjtjResidualwise(psad,residualgroup,fnkind), PCGStep1Wrapper)
            end
        elseif A.PRECOMPUTE_JTJ:isclassof(jtjp) then
            local precomputeJtJWrapper = computeSimpleResidualWrapper(residual_cardinality, false, 2)
            local materializeJtJ = creatematerializejtjResidualwise(psad,residualgroup,fnkind)
            addFunctionSpec(nonlinear_kernels, "precomputeJtJ", materializeJtJ, precomputeJtJWrapper)
        elseif A.APPLY_SEPARATELY:isclassof(jtjp) then
            -- linear_kernels["PCGStep1_J"] = A.KernelCostData(unknown_cardinality, 2, 0, 0, 1)
            -- linear_kernels["PCGStep1_Jt"] = A.KernelCostData(unknown_cardinality, 2, 0, 0, 1)
            local PCGStep1_JWrapper = computeSimpleResidualWrapper(full_cardinality, false, 2)
            addFunctionSpec(linear_kernels, "PCGStep1_J", createapplyjResidualwise(psad,residualgroup,full_fnkind), PCGStep1_JWrapper)
            local do_reduce = not multistep_alphaDenominator_compute
            local PCGStep1_JtWrapper = computeSimpleResidualWrapper(full_cardinality, do_reduce, 2)
            addFunctionSpec(linear_kernels, "PCGStep1_Jt", createapplyjtResidualwise(psad,residualgroup,full_fnkind), PCGStep1_JtWrapper)
        elseif A.PRECOMPUTE_J:isclassof(jtjp) or A.PRECOMPUTE_J_THEN_JTJ:isclassof(jtjp) then
            -- PrecomputeJ
            local ops_per_partial = 13 -- Check
            local writes_per_partial = 2
            local extra_pointers = 4
            local row_ptr_write = 1
            local rg_domain = residualgroup.domainandschedule.domain
            local inner_domains = util.list_subtraction(rg_domain.full.domains, rg_domain.external.domains)
            if #inner_domains > 0 then
                local threads_per_residuial = 1
                for _,d in ipairs(inner_domains) do
                    threads_per_residuial = threads_per_residuial * d.dim.size
                end
                row_ptr_write = 1 / threads_per_residuial
            end

            local computeJ = createcomputejResidualwise(psad,residualgroup,full_fnkind)
            local num_partials = #computeJ.results
            local reg_count = 5 + extra_pointers * 2 + num_partials
            local op_count = 5 + ops_per_partial * num_partials
            local memwrites = writes_per_partial * num_partials + row_ptr_write
            local PrecomputeJWrapper = A.KernelCostData(full_cardinality, 
                                                        reg_count, -- register count 
                                                        op_count, -- op count
                                                        0, -- mem_read_count
                                                        memwrites) -- mem_write_count
            addFunctionSpec(nonlinear_kernels, "PrecomputeJ", computeJ, PrecomputeJWrapper)
        else
            assert("Invalid jtjp schedule for residual group "..i)
        end

        if psad.P:UsesLambda() then
            local PCGComputeCtCWrapper = computeSimpleResidualWrapper(full_cardinality, false, 1)
            addFunctionSpec(nonlinear_kernels, "PCGComputeCtC", createjtfResidualwise(psad,residualgroup,full_fnkind), PCGComputeCtCWrapper)

            local ModelCostWrapper = computeSimpleResidualWrapper(residual_cardinality, true, 1)
            addFunctionSpec(nonlinear_kernels, "ModelCost", createmodelcostResidualwise(psad,residualgroup,fnkind), ModelCostWrapper) 
        end
    end

    if not scheduledenergy:CanFusePCGInit() then
        local params = {
            kernel_table=nonlinear_kernels, base_name="PCGInit1_Finish", 
            unknown_variable_count=3, scalar_variable_count=1, pointer_variable_count=4,
            unknown_ops=2, unknown_to_scalar_ops=1, scalar_ops=0, 
            unknown_mem_reads=1, scalar_mem_reads=0, 
            unknown_mem_writes=2, scalar_mem_writes=0,
            scalar_reductions=1
        }
        if usepreconditioner then
            params.pointer_variable_count = params.pointer_variable_count + 1
            params.unknown_mem_reads = params.unknown_mem_reads + 1
            params.unknown_ops = params.unknown_ops + guarded_invert_opcount
        end
        addUnknownwiseKernels(params)
    end

    if scheduledenergy:RequiresJ() then
        if not materialized_is_dense then
            -- TODO: more accutate estimates
            -- print("Check for performance of cusparse kernels, potentially make new overhead base")
            nonlinear_kernels["cusparseCreateIdentityPermutation"] = A.KernelCostData(
                                                        nnzMaterialized, -- threadcount
                                                        8, -- register count 
                                                        8, -- op count
                                                        0, -- mem_read_count
                                                        1) -- mem_write_count

            local partials_per_residual = nnzMaterialized/nResidualsMaterialized
            -- Assume one thread per row, n2 sort in 
            -- Have to read permutation vector and colInd (and row pointer bounds)
            local csrSort_costdata = A.KernelCostData(nResidualsMaterialized, -- threadcount
                                                        8+partials_per_residual, -- register count 
                                                        8+partials_per_residual*partials_per_residual, -- op count
                                                        (2+2*partials_per_residual)*util.cu.uncoalesced_multiplier, -- mem_read_count
                                                        2*partials_per_residual*util.cu.uncoalesced_multiplier) -- mem_write_count

            local csrGather_costdata = A.KernelCostData(nnzMaterialized, -- threadcount
                                                        8, -- register count 
                                                        8, -- op count
                                                        2*util.cu.uncoalesced_multiplier, -- mem_read_count
                                                        1) -- mem_write_count

            nonlinear_kernels["cusparseXcsrsort"] = csrSort_costdata
            -- step 4: gather sorted csrVal
            nonlinear_kernels["cusparseSgthr"] = csrGather_costdata

            --[[
                The implementation of cusparseScsr2csc is not open knowledge.
                For ease of calculation, we'll just use algorithm 2 in 
                https://kaixih.github.io/assets/papers/wang-transposition-ics16.pdf
                which is five-phase: 
                1. Atomic construction of ColPtr and dloc (dloc is size nnz)
                2. Prefix sum of ColPtr
                3. Write RowIdx, scatter cscVal
                4. Sort 
                5. Gather sorted
            --]]
            nonlinear_kernels["cusparseScsr2csc_1"] = A.KernelCostData(
                                                        nResidualsMaterialized, -- threadcount
                                                        8, -- register count 
                                                        8+3*partials_per_residual, -- op count
                                                        2+(1+util.cu.uncoalesced_multiplier)*partials_per_residual, -- mem_read_count
                                                        (1+util.cu.uncoalesced_multiplier)*partials_per_residual) -- mem_write_count
            -- Prefix sum; it is more complicated, but we can approximate as if serial...
            nonlinear_kernels["cusparseScsr2csc_2"] = A.KernelCostData(
                                                        nnzMaterialized, -- threadcount
                                                        8, -- register count 
                                                        8, -- op count
                                                        1, -- mem_read_count
                                                        1) -- mem_write_count
            nonlinear_kernels["cusparseScsr2csc_3"] =  A.KernelCostData(
                                                        nResidualsMaterialized, -- threadcount
                                                        16, -- register count 
                                                        8+2*partials_per_residual, -- op count
                                                        2+(3+util.cu.uncoalesced_multiplier)*partials_per_residual, -- mem_read_count
                                                        2*util.cu.uncoalesced_multiplier*partials_per_residual) -- mem_write_count
            nonlinear_kernels["cusparseScsr2csc_4"] = csrSort_costdata
            nonlinear_kernels["cusparseScsr2csc_5"] = csrGather_costdata
        end
        if scheduledenergy:RequiresMatMul() then
             if materialized_is_dense then
                --[[
                for (i=0;i<N;i++)
                    for (j=0;j<N;j++)
                        C[i,j] = 0;
                        for (k=0;k<R;k++)
                            C[i,j] += A[i,k]*B[k,j];
                
                Assume each read is shared between 32 threads?
                TODO: empirical estimation?
                --]]
                nonlinear_kernels["cublasSgemm_v2"] = A.KernelCostData(unknown_cardinality*unknown_cardinality, 
                    16, 
                    2*nResidualsMaterialized, 
                    nResidualsMaterialized*(2.0/32), 
                    1) 
            else
                --[[
                    For estimation's sake, we'll just assume an easy to understand kernel
                    One thread per JtJ entry. Each thread must read all partials contributing to it:
                    two per residual contributing to it (one on diagonals, but we'll ignore that complication)
                    the indices for the contributing columns (lets ignore the row pointer for now)
                    multiply them together, accumulate a sum, and write to an entry specified by row pointers and such
                    
                --]]
                local estimated_contributing_pairs_per_entry = nResidualsMaterialized*density

                --local unknowns_per_residual = unknown_cardinality / residual_cardinality
                nonlinear_kernels["cusparseScsrgemm"] =  A.KernelCostData(estimatedNnzJtJ, 16, 8+3*estimated_contributing_pairs_per_entry, 
                    (8*4+(util.cu.uncoalesced_multiplier*2+2)*2*estimated_contributing_pairs_per_entry), 1)  
            end
        end
    end
    if scheduledenergy:RequiresJtJMaterialize() then
        nonlinear_kernels["JTJClear"] = A.KernelCostData(unknown_cardinality*unknown_cardinality, 2, 0, 0, 1)
    end
    if uses_lambda then
        nonlinear_kernels["CtC_clear"] = unknownwise_clear_kernel 

        -- Assuming JacobiScalingType.ONCE_PER_SOLVE; TODO: read programmatically
        addUnknownwiseKernels{
            kernel_table=nonlinear_kernels, base_name="PCGFinalizeDiagonal", 
            unknown_variable_count=6, scalar_variable_count=2, pointer_variable_count=10,
            unknown_ops=18, unknown_to_scalar_ops=2, scalar_ops=1, 
            unknown_mem_reads=3, scalar_mem_reads=3, 
            unknown_mem_writes=4, scalar_mem_writes=0,
            scalar_reductions=2
        }
    end
    
    if uses_lambda then
        nonlinear_kernels["savePreviousUnknowns"] = unknownwise_copy_kernel
    end

    nonlinear_kernels["PCGLinearUpdate"] = A.KernelCostData(unknown_cardinality, 8, 1, 2, 1)

   
    --nonlinear_kernels["precompute"] = A.KernelCostData(unknown_cardinality, 2, 0, 0, 1)
    addFunctionSpecs(nonlinear_kernels, createprecomputed(psad.precomputed))
    addFunctionSpecs(nonlinear_kernels, nonlin_functionspecs)


    -------------------- LINEAR KERNELS

    -- KernelCostData = (number thread_count, number register_count, number opcount_per_thread, number memreads_per_thread, number memwrites_per_thread)
    -- TODO: Test multi-row kernels
    local function computeDenseMatrixKernelCostData(kernel_store, base_name, numrows, numcols)
        local rowwiseDenseMatVec     = A.KernelCostData(numrows, 8, numcols*2, numcols*2, 1)
        local elementwiseDenseMatVec = A.KernelCostData(numrows*numcols, 8, 2, 2, 1/util.cu.warpSize)
        if cost_of_kernel(elementwiseDenseMatVec) < cost_of_kernel(rowwiseDenseMatVec) then
            kernel_store[base_name.."_elementwise"] = elementwiseDenseMatVec
        else
            kernel_store[base_name.."_rowwise"] = rowwiseDenseMatVec
        end
    end

    --[[
        int row = blockDim.x * blockIdx.x + threadIdx.x ;
        if( row < num_rows ){
            float dot = 0;
            int row_start = ptr [ row ];
            int row_end = ptr [ row +1];
            for (int jj = row_start ; jj < row_end ; jj ++)
                dot += data [ jj ] * x[ indices [ jj ] ];
            y[ row ] += dot ;
        }
    --]]
    local function computeSparseMatrixKernelCostData(kernel_store, base_name, numrows, numcols, nnz)
        -- Using the kernel in figure 20 of https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf
        -- Since we don't take into account overread from non-contiguous memory access, this should be fine
        -- approximation uncoalesced_multiplier()
        local entries_per_row = nnz/numrows
        kernel_store[base_name] = A.KernelCostData(numrows, 16, 8+5+(6*entries_per_row), 3+((2+1*util.cu.uncoalesced_multiplier)*entries_per_row), 1)
    end

    
    local needsSummation = false
    if scheduledenergy:RequiresJtJMaterialize() then
        computeDenseMatrixKernelCostData(linear_kernels, "cublas_JTJ_p", unknown_cardinality, unknown_cardinality)
        needsSummation = true
    else
        if do_clears then
            linear_kernels["Ap_X_clear"] = unknownwise_clear_kernel
        end
    end
    if scheduledenergy:RequiresJ() then
        if scheduledenergy:RequiresMatMul() then
            if materialized_is_dense then
                computeDenseMatrixKernelCostData(linear_kernels, "cublas_JTJ_p", unknown_cardinality, unknown_cardinality)
            else
                computeSparseMatrixKernelCostData(linear_kernels, "cusparseScsrmv_JtJ_p", unknown_cardinality, 
                                                    unknown_cardinality, estimatedNnzJtJ)
            end
        else
            linear_kernels["cusparse_Jp_clear"] = A.KernelCostData(residual_cardinality, 8, 0, 0, 1) 
            if materialized_is_dense then
                computeDenseMatrixKernelCostData(linear_kernels, "cublas_J_p", residual_cardinality, unknown_cardinality)
                computeDenseMatrixKernelCostData(linear_kernels, "cublas_JT_Jp", unknown_cardinality, residual_cardinality)
            else
                computeSparseMatrixKernelCostData(linear_kernels, "cusparseScsrmv_J_p", residual_cardinality, unknown_cardinality, nnzMaterialized)
                computeSparseMatrixKernelCostData(linear_kernels, "cusparseScsrmv_JT_Jp", unknown_cardinality, residual_cardinality, nnzMaterialized)
            end
        end
    end
    --if scheduledenergy:RequiresApplyJtJp() then
        -- PCGStep1
    --end
    if scheduledenergy:RequiresSeparateJtAndJ() then
        linear_kernels["Jp_clear"] = A.KernelCostData(residual_cardinality, 8, 0, 0, 1)
        -- Split PCGStep1
    end

    if multistep_alphaDenominator_compute then
        local params = {
            kernel_table=linear_kernels, base_name="PCGStep1_Finish", 
            unknown_variable_count=2, scalar_variable_count=1, pointer_variable_count=3,
            unknown_ops=1, unknown_to_scalar_ops=1, scalar_ops=0,
            unknown_mem_reads=2, scalar_mem_reads=0, 
            unknown_mem_writes=0, scalar_mem_writes=0,
            scalar_reductions=1
        }
        addUnknownwiseKernels(params)
    end

    do
        local params = {
            kernel_table=linear_kernels, base_name="PCGStep2", 
            unknown_variable_count=7, scalar_variable_count=4, pointer_variable_count=8,
            unknown_ops=6, unknown_to_scalar_ops=1, scalar_ops=4,
            unknown_mem_reads=4, scalar_mem_reads=2, 
            unknown_mem_writes=3, scalar_mem_writes=0,
            scalar_reductions=1
        }
        if usepreconditioner then
            params.pointer_variable_count = params.pointer_variable_count + 1
            params.unknown_variable_count = params.unknown_variable_count + 1
            params.unknown_ops = params.unknown_ops + 1
            params.unknown_mem_reads = params.unknown_mem_reads + 1
        end

        if uses_lambda then
            params.unknown_variable_count = params.unknown_variable_count + 1
            params.scalar_variable_count = params.scalar_variable_count + 1
            params.pointer_variable_count = params.pointer_variable_count + 1
            params.unknown_ops = params.unknown_ops + 2
            params.unknown_to_scalar_ops = params.unknown_to_scalar_ops + 1
            params.scalar_ops = params.scalar_ops + 1
            params.unknown_mem_reads = params.unknown_mem_reads + 1
            params.scalar_reductions = params.scalar_reductions + 1
        end
        addUnknownwiseKernels(params)
    end
    do
        local params = {
            kernel_table=linear_kernels, base_name="PCGStep3", 
            unknown_variable_count=2, scalar_variable_count=3, pointer_variable_count=4,
            unknown_ops=2, unknown_to_scalar_ops=0, scalar_ops=4,
            unknown_mem_reads=2, scalar_mem_reads=2, 
            unknown_mem_writes=1, scalar_mem_writes=0,
            scalar_reductions=0
        }
        addUnknownwiseKernels(params)
    end

    addFunctionSpecs(linear_kernels, lin_functionspecs)

    -- Nonlinear term: \sum_{k_{N_i} \in K_N} C(k_{N_i})
    local nonlinear_cost = 0
    for name,kernel in pairs(nonlinear_kernels) do
        local kernel_cost = cost_of_kernel(kernel)
        if verbose then
            print(name..", "..tostring(kernel_cost))
            --print(kernel)
        end
        nonlinear_cost = nonlinear_cost + kernel_cost
    end

    local linear_cost = 0
    for name,kernel in pairs(linear_kernels) do
        local kernel_cost = cost_of_kernel(kernel)
        if verbose then
            print(name..", "..tostring(kernel_cost))
            --print(kernel)
        end
        linear_cost = linear_cost + kernel_cost
    end

    print("Total Nonlinear Cost, "..tostring(nonlinear_cost))
    print("Total Linear Cost, "..tostring(linear_cost))
    local total_cost = (nonlinear_cost + lin_iter_hint*linear_cost)
    print("Total Cost, "..tostring(total_cost))
    return total_cost
end

local function cost_of_schedule(psad,energy,schedule,lin_iter_hint,verbose)
    apply_schedule(energy,schedule)
    local scheduledenergy = A.ScheduledEnergy(toresidualgroups(energy))
    return cost_of_scheduled_energy(psad,scheduledenergy,lin_iter_hint,verbose)
end


local function heuristic_autoschedule(psad, energy, unknown_count, lin_iter_hint, autoschedule_index)

    -- Specify each step as a separate function up here, so the code reflects the high level description from the paper.
    local function clear_scheduling_directives(energy)
        apply_schedule(energy,default_schedule(energy))
    end

    local function aggressively_merge_energy(energy)
        -- Aggressively merge residuals mapped over the same domains
        -- For loop bounds are only calculated once, so use raw while
        local i = 1
        while i <= #energy.residuals do 
            local r1 = energy.residuals[i]
            local j = i + 1
            while j <= #energy.residuals do
                local r2 = energy.residuals[j]
                if A.IterationDomain(r1.domains) == A.IterationDomain(r2.domains) then
                    energy:merge(r1,r2)
                else 
                    j = j + 1 -- if we merge, then the residual at j has changed, so "re-run" at that value of j
                end
            end
            i = i + 1
        end
    end

    local function select_expressions_to_materialize(energy,psad,lin_iter_hint)
        local materialized_expressions = List()
        for k,_ in pairs(ComputedArrayCache) do
            materialized_expressions:insert(k)
        end
        materialized_expressions:sort(function(e1,e2)
            return ComputedArrayCache[e1].computed_array_index < ComputedArrayCache[e2].computed_array_index
        end)

        for _,exp in ipairs(materialized_expressions) do
            exp:set_materialize(true)
        end

        local inlined_exp_count = 0
        local inlined_grad_count = 0
        for _,exp in ipairs(materialized_expressions) do
            local lowest_cost = cost_of_scheduled_energy(psad,A.ScheduledEnergy(toresidualgroups(energy)),lin_iter_hint,false)
            exp:set_materialize(false)
            local new_cost = cost_of_scheduled_energy(psad,A.ScheduledEnergy(toresidualgroups(energy)),lin_iter_hint,false)
            if new_cost < lowest_cost then
                local rel_diff = (lowest_cost - new_cost) / lowest_cost
                print("Decrease of "..tostring(rel_diff*100.0).."% by inlining")
                lowest_cost = new_cost
                inlined_exp_count = inlined_exp_count + 1
            else
                exp:set_materialize(true)
            end
            exp:set_gradient_materialize(false)
            new_cost = cost_of_scheduled_energy(psad,A.ScheduledEnergy(toresidualgroups(energy)),lin_iter_hint,false)
            if new_cost < lowest_cost then
                local rel_diff = (lowest_cost - new_cost) / lowest_cost
                print("Decrease of "..tostring(rel_diff*100.0).."% by inlining grad")
                lowest_cost = new_cost
                inlined_grad_count = inlined_grad_count + 1
            else
                exp:set_gradient_materialize(true)
            end
        end
        print("Inlined "..tostring(inlined_exp_count).." exps and "..tostring(inlined_grad_count).." gradients, out of a possible "..tostring(#materialized_expressions))
    end

    local function select_jtjp_materialization(energy,psad,lin_iter_hint)
        for i,rg in ipairs(energy.residuals) do
            local materializej = {false,true,true}
            local materializejp = {true,true,false}
            local materializejtj = {false,false,true}
            local function add_to_materialize_lists(j,jp,jtj)
                materializej[#materializej+1] = j
                materializejp[#materializejp+1] = jp
                materializejtj[#materializejtj+1] = jtj
            end
            if not has_reduction(rg) then
                add_to_materialize_lists(false,false,false)
                if #psad.P:UnknownType():IndexSpaces() == 1 then
                    add_to_materialize_lists(false,false,true)
                end
            end

            local function set_jtjp_materialization(i)
                rg.J:set_materialize(materializej[i])
                rg.Jp:set_materialize(materializejp[i])
                rg.JtJ:set_materialize(materializejtj[i])
            end

            local lowest_cost = math.huge
            local index_of_lowest_cost = 1
            for i=1,#materializej do
                set_jtjp_materialization(i)
                print("Testing Materialization for "..rg.name.." J="..tostring(materializej[i]).." Jp="..tostring(materializejp[i]).." JtJ="..tostring(materializejtj[i]))
                local scheduledenergy = A.ScheduledEnergy(toresidualgroups(energy))
                local estimated_cost = cost_of_scheduled_energy(psad,scheduledenergy,lin_iter_hint,false)
                if estimated_cost < lowest_cost then
                    index_of_lowest_cost = i
                    lowest_cost = estimated_cost
                end
            end
            print("Selected Materialization for "..rg.name.." J="..tostring(materializej[index_of_lowest_cost]).." Jp="..tostring(materializejp[index_of_lowest_cost]).." JtJ="..tostring(materializejtj[index_of_lowest_cost]))
            set_jtjp_materialization(index_of_lowest_cost)
        end
    end

    local function choose_compute_at_output(energy)
        local compute_at_output = true
        for i,rg in ipairs(energy.residuals) do
            rg:compute_at_output(false)
            local function use_compute_at_output(rg)
               if not (rg.JtJ.materialize or rg.J.materialize or rg.Jp.materialize) then
                    local domains,domains_to_templates = MapAndGroupBy(rg.expressions,getclassifyexpression(rg:domain_order()))
                    for _,domain in ipairs(domains) do
                        for _,template in ipairs(domains_to_templates[domain]) do
                            for _,u in ipairs(template.unknowns) do
                                local dims = u.image.type.ispace.dims
                                if #domain.full.domains ~= #dims then
                                    return false
                                end
                                for i,d in ipairs(dims) do
                                    if d ~= domain.full.domains[i].dim then
                                        return false
                                    end
                                end
                            end
                        end
                    end
                    return true
                end
                return false
            end
            compute_at_output = compute_at_output and use_compute_at_output(rg)
        end

        for _,rg in ipairs(energy.residuals) do
            rg:compute_at_output(compute_at_output)
            rg.JtF:compute_at_output(compute_at_output)
        end
    end

    local function reorder_for_coherence(res)
        -- We sort the indexing order of the residuals such that any Index Domain used in the residual that 
        -- is not used to access an unknown is brought to the front (so it is the innermost iterator);
        res:clear_reorder()
        local unknowns = List()
        for _,exp in ipairs(res.expressions) do
            unknowns:insertall(extract_unknowns(exp))
        end
        local unknown_domains = get_index_domains_from_unknown_list(unknowns)
        local unused_in_unk = util.list_subtraction(res.domains,unknown_domains)
        local used_in_unk = util.list_subtraction(res.domains,unused_in_unk)
        
        local new_ordering = List()
        new_ordering:insertall(unused_in_unk)
        new_ordering:insertall(used_in_unk)
        
        res:reorder(new_ordering)
    end


    -- It's possible the user added scheduling directives before calling the autoscheduler. Clear them out beforehand.
    clear_scheduling_directives(energy)

    --1. The autoscheduler aggressively merges all residual groups that are mapped over the same Index Domains (this monotonically reduces the cost in the cost model if there are any shared terms, otherwise it is cost-neutral).
    aggressively_merge_energy(energy)
    -- Create default schedules to evaluate
    -- Set "Default" schedule
    for i,rg in ipairs(energy.residuals) do
        rg.J:set_materialize(false)
        rg.Jp:set_materialize(true)
        rg.JtJ:set_materialize(false)
    end

    --3. We insert materializations whenever reading an expression and its Jacobian would lower cost compared to computing the original expression (including the new materialization kernel in the cost calculation).
    select_expressions_to_materialize(energy, psad, lin_iter_hint)
    
    --4. For the merged groups, we choose the high-level $\jtjp$ materialization strategy by evaluating the costs over the five combinations.
    select_jtjp_materialization(energy, psad, lin_iter_hint)

    --5. If the schedule for a group is $\jtjp$ and the residuals are mapped over the dimensions of the unknowns, we map over elements of the output (this does not require a potentially expensive inverse map, and lowers the number of required memory writes).
    choose_compute_at_output(energy)

    --6. We sort the indexing order of the residuals such that any Index Domain used in the residual that is not used to access an unknown is brought to the front (so it is the innermost iterator); this is a mechanical process that, combined with coherent reduction code emitted by our compiler, drastically improves performance, see Figure \ref{fig:reorder}.
    for i,rg in ipairs(energy.residuals) do
        reorder_for_coherence(rg)
    end
    return true
end

local function autoschedule(psad, energy, unknown_count, lin_iter_hint, autoschedule_index)
    if _thallo_use_autoscheduler == 1 and use_new_autoscheduler then
        return heuristic_autoschedule(psad, energy, unknown_count, lin_iter_hint, autoschedule_index)
    end

    if _thallo_use_autoscheduler == 1 then -- Old-style autoscheduler
        local compute_at_output = true
        for i,rg in ipairs(energy.residuals) do
            local name = rg.name
            if has_reduction(rg) then
                print("Autoscheduler setting Jt[Jp]")
                rg.Jp:set_materialize(true)
                rg.JtJ:set_materialize(false)
                rg.J:set_materialize(false)
            else
                if unknown_count < dense_materialize_threshold then
                    print("Autoscheduler setting [JtJ]p")
                    rg.JtJ:set_materialize(true)
                    rg.Jp:set_materialize(false)
                    rg.J:set_materialize(false)
                else
                    print("Autoscheduler setting JtJp")
                    -- don't materialize anything
                    rg.JtJ:set_materialize(false)
                    rg.Jp:set_materialize(false)
                    rg.J:set_materialize(false)
                end
            end
            rg:clear_reorder()
            rg._compute_at_output = false
            -- TODO: auto reorder indices to get coherent reductions if possible
            -- TODO: If necessary, detect indexed expressions and precompute
            local function use_compute_at_output(rg)
               if not (rg.JtJ.materialize or rg.J.materialize or rg.Jp.materialize) then
                    local domains,domains_to_templates = MapAndGroupBy(rg.expressions,getclassifyexpression(rg:domain_order()))
                    for _,domain in ipairs(domains) do
                        for _,template in ipairs(domains_to_templates[domain]) do
                            for _,u in ipairs(template.unknowns) do
                                local dims = u.image.type.ispace.dims
                                if #domain.full.domains ~= #dims then
                                    return false
                                end
                                for i,d in ipairs(dims) do
                                    if d ~= domain.full.domains[i].dim then
                                        return false
                                    end
                                end
                            end
                        end
                    end
                    return true
                end
                return false
            end
            compute_at_output = compute_at_output and use_compute_at_output(rg)
        end
        for _,rg in ipairs(energy.residuals) do
            rg:compute_at_output(compute_at_output)
            rg.JtF:compute_at_output(compute_at_output)
        end
    elseif _thallo_use_autoscheduler == 2 or ((_thallo_use_autoscheduler > 2 or _thallo_use_autoscheduler == -1) and autoschedule_index < 0) then
        -- Clear everything
        for i,rg in ipairs(energy.residuals) do
            rg:clear_reorder()
            rg._compute_at_output = false
            -- don't materialize anything
            rg.JtJ:set_materialize(false)
            rg.Jp:set_materialize(has_reduction(rg))
            rg.J:set_materialize(false)
        end
    else -- Exhaustive Autoschedule
        local too_many_schedules = 2000
        local stochastic_sample_size = 1000
        local all_schedules, schedule_count = generate_all_schedules(energy, unknown_count)
        print("Schedule Count")
        print(#all_schedules)

        if #all_schedules < stochastic_sample_size and not limited_search_space then
            normalize_energy(energy)
            all_schedules, schedule_count = generate_all_schedules(energy, unknown_count)
            print("Normalized Schedule Count")
            print(#all_schedules)
        end

        local function find_min_cost(all_schedules,psad,energy,lin_iter_hint,verbose)
            local min_cost = 9999999999999999999.9
            local index = 1
            for i,s in ipairs(all_schedules) do
                local cost = cost_of_schedule(psad, energy, s, lin_iter_hint, false)
                if verbose then
                    print("Schedule "..tostring(i-1).."/"..tostring(all_schedules)..": ")
                    print(s)
                    print("Estimated Cost:")
                    print(cost)
                end
                if cost < min_cost then
                    min_cost = cost
                    index = i
                end
            end
            return index,min_cost
        end

        if #all_schedules > too_many_schedules then
            math.randomseed(8934723) -- Deterministic sample 
            local sum = 0
            for i=1,100 do sum = sum + math.random() end
            print("RNG tag: "..tostring(sum)) -- Record output to check nothing weirds happening with random
            local schedule_sample_indices = {}
            local sampled_schedule_count = 0

            --[[
            -- Make sure the min cost is in the sample
            local index, min_cost = find_min_cost(all_schedules,psad,energy,lin_iter_hint,true)
            schedule_sample_indices[index] = true
            sampled_schedule_count = sampled_schedule_count + 1
            --]]
            while sampled_schedule_count < stochastic_sample_size do
                local new_idx = math.random(1, #all_schedules)
                if not schedule_sample_indices[new_idx] then
                    schedule_sample_indices[new_idx] = true
                    sampled_schedule_count = sampled_schedule_count + 1
                end
            end

            -- Sort into ascending order
            local sorted_indices = List()
            for k,_ in pairs(schedule_sample_indices) do
                sorted_indices:insert(k)
            end
            sorted_indices:sort()

            local sampled_schedules = List()
            for _,i in ipairs(sorted_indices) do
                sampled_schedules:insert(all_schedules[i])
                print(i)
            end
            all_schedules = sampled_schedules
        end
        print("New Schedule Count")
        print(#all_schedules)

        local schedule
        if _thallo_use_autoscheduler < -1 then -- "Perfect" Autoschedule
            local index, min_cost = find_min_cost(all_schedules,psad,energy,lin_iter_hint,true)
            schedule = all_schedules[index]
        elseif _thallo_use_autoscheduler == -1 then
            -- Dump schedules, undocumented "feature"
            for i=1,#all_schedules do
                schedule = all_schedules[i]
                logSchedule(schedule)
                local cost = cost_of_schedule(psad, energy, schedule, lin_iter_hint,true)
                writeScheduleResults_lua(0.0, 0.0, i-1, cost)
                print("Writing Schedule:"..i)
            end
            print("intentionally crashing")
            assert(false)
        else
            if autoschedule_index >= #all_schedules then
                return false
            end
            schedule = all_schedules[autoschedule_index+1]
            print("Schedule "..tostring(autoschedule_index)..": ")
            print(schedule)
            print("Lin Iter Hint: "..lin_iter_hint)
            local cost = cost_of_schedule(psad, energy, schedule, lin_iter_hint, true)
            psad.P.estimated_cost = cost
            print("Estimated Cost: "..tostring(cost))
        end
        apply_schedule(energy,schedule)
        if autoschedule_index >= 0 then
            logSchedule(schedule)
        end
    end
    return true
end

function A.ProblemSpecAD:getexcludes()
    for _,ip in ipairs(self.P:UnknownType().images) do
        self.excludeexps:insertall(self.nametoimage[ip.name].excludeexps)
    end
    self.excludeexps = util.uniquify(self.excludeexps)
end

function A.ProblemSpecAD:Cost(energy,exauto_index,lin_iter_hint)
    self.P.estimated_cost = 0.0
    local functionspecs = List()
    self:getexcludes()
    if not lin_iter_hint or lin_iter_hint <= 0 then
        lin_iter_hint = 10
    end
    if _thallo_use_autoscheduler ~= 0 then
        local numunknowns = self.P:UnknownType():cardinality()
        if not autoschedule(self, energy, numunknowns, lin_iter_hint, exauto_index) then
            return nil
        end
    end
    local scheduled_energy = A.ScheduledEnergy(toresidualgroups(energy))
    local cost = cost_of_scheduled_energy(self, scheduled_energy, lin_iter_hint, false)
    print("Cost: "..tostring(cost))
    --assert(false)
    self:SetScheduledEnergy(scheduled_energy)
    
    self.P._direct_solve = energy._direct_solve

    for i,residualgroup in ipairs(self.P.scheduledenergy.residualgroups) do
        local schedule = residualgroup.domainandschedule.schedule
        local is_unknown_schedule = A.AtOutputSchedule:isclassof(schedule.fnschedule)
        local is_unknown_jtf_schedule = A.AtOutputSchedule:isclassof(schedule.jtfschedule)
        local fnschedule = A.ResidualSchedule
        local domain = residualgroup.domainandschedule.domain
        
        local fnkind = A.ResidualwiseFunction(domain.external,fnschedule)
        fnkind.residualindex = A.ImageIndex(domain.external.domains:map(function(dom)
            return A.DirectIndexComponent(dom)
        end))
        fnkind.name  = "residual_"..tostring(i)
        local full_fnkind = fnkind
        if domain.external ~= domain.full then
            full_fnkind = A.ResidualAndContractionwiseFunction(domain.full,fnschedule)
            full_fnkind.residualindex = A.ImageIndex(domain.external.domains:map(function(dom)
                return A.DirectIndexComponent(dom)
            end))
            full_fnkind.name = fnkind.name
        end

        functionspecs:insert(createcost(residualgroup,fnkind))

        local jtjp = schedule.jtjpschedule

        --jtf
        if is_unknown_jtf_schedule then
            local unknownwise_fnkind = A.UnknownwiseFunction(fnkind.domain)
            functionspecs:insert(createjtfcentered(self,residualgroup,unknownwise_fnkind))
        else
            functionspecs:insert(createjtfResidualwise(self,residualgroup,full_fnkind))
        end

        --jtjp
        if A.INLINE:isclassof(jtjp) then
            if is_unknown_schedule then
                local unknownwise_fnkind = A.UnknownwiseFunction(fnkind.domain)
                functionspecs:insert(createjtjcentered(self,residualgroup,unknownwise_fnkind))
            else
                functionspecs:insert(createapplyjtjResidualwise(self,residualgroup,fnkind))
            end
        elseif A.PRECOMPUTE_JTJ:isclassof(jtjp) then
            functionspecs:insert(creatematerializejtjResidualwise(self,residualgroup,fnkind))
        elseif A.APPLY_SEPARATELY:isclassof(jtjp) then
            functionspecs:insert(createapplyjResidualwise(self,residualgroup,full_fnkind))
            functionspecs:insert(createapplyjtResidualwise(self,residualgroup,full_fnkind))
        elseif A.PRECOMPUTE_J:isclassof(jtjp) or A.PRECOMPUTE_J_THEN_JTJ:isclassof(jtjp) then
            functionspecs:insert(createcomputejResidualwise(self,residualgroup,full_fnkind))
        else
            assert("Invalid jtjp schedule for residual group "..i)
        end

        if self.P:UsesLambda() then
            functionspecs:insert(computeCtCResidualwise(self,residualgroup,fnkind))
            functionspecs:insert(createmodelcostResidualwise(self,residualgroup,fnkind))         
        end
        for i,exclude in ipairs(self.excludeexps) do
            -- TODO: better error checking here
            local unk_dom = get_unknown_iteration_domain(residualgroup.residuals[1].unknowns)
            local unknownwise_fnkind = A.UnknownwiseFunction(unk_dom)
            local class = getclassifyexpression(domain.full.domains)(exclude)
            functionspecs:insert(A.FunctionSpec(unknownwise_fnkind, "exclude", EMPTY, List{exclude}, EMPTY))
        end
    end
    functionspecs:insertall(createprecomputed(self.precomputed))
    
    self:AddFunctions(functionspecs)
    
    return self.P
end


function A.Energy:set_direct_solve(v)
    self._direct_solve = v
end

local function get_flat_residual_list(rg)
    local Rs = terralib.newlist {}
    for _,e in ipairs(rg) do
        if ad.ExpVector:isclassof(e) then
            for i,t in ipairs(e:expressions()) do
                t = assert(ad.toexp(t), "expected an ad expression")
                Rs:insert(t)
            end
        else
            Rs:insert((assert(ad.toexp(e), "expected an ad expression")))
        end
    end
    return Rs
end

function A.Energy:init()
    self._direct_solve = false
    for _,res in ipairs(self.residuals) do
        assert(not self[res.name], "Cannot name residual "..res.name..". It causes a name conflict.")
        self[res.name] = res
    end
end

function A.NamedResidual:domain_order()
    return self._domain_order
end

function A.NamedResidual:reorder(domains)
    self._domain_order = List()
    for i,d in ipairs(domains) do
        assert(A.IndexDomain:isclassof(d), "Invalid domain in reorder statement; Make sure to use Index Domains, not Dimensions")
        self._domain_order:insert(d)
    end
    return self
end

function A.NamedResidual:clear_reorder()
    self._domain_order = nil
end

function A.Energy:merge(r1,r2)
    assert(A.IterationDomain(r1.domains) == A.IterationDomain(r2.domains))
    local i1 = table.indexOf(self.residuals,r1)
    local i2 = table.indexOf(self.residuals,r2)
    assert(i1 > 0)
    assert(i2 > 0)
    self.residuals:remove(i2)
    r1.name = r1.name.."_"..r2.name
    r1.expressions:insertall(r2.expressions)
    r2.invalid = true
    return r1
end

function A.Energy:split(r1,start,finish)
    local i1 = table.indexOf(self.residuals,r1)
    assert(i1 > 0)
    assert(start >= 1 and start <= #r1.expressions)
    assert(finish >= 1 and finish <= #r1.expressions and start <= finish)
    local inner_expressions = List()
    for i=finish,start,-1 do
        inner_expressions:insert(r1.expressions:remove(i))
    end
    local new_name = r1.name.."_"..tostring(start)
    if start ~= finish then
        new_name = new_name.."_"..tostring(finish)
    end
    local r2 = A.NamedResidual(new_name, inner_expressions)
    r2:assume_schedule(r1)
    self.residuals:insert(r2)
    self[new_name] = r2
    return r2,r1 
end

function A.Energy:full_split(r)
    if #r.expressions > 1 then
        local i_orig = table.indexOf(self.residuals,r)
        assert(i_orig > 0)
        for i,exp in ipairs(r.expressions) do
            local new_name = r.name.."_"..tostring(i)
            local new_exps = List()
            new_exps:insert(exp)
            local r_new = A.NamedResidual(new_name, new_exps)
            r_new:assume_schedule(r)
            self.residuals:insert(r_new)
            self[new_name] = r_new
        end
        self[r.name] = nil
        self.residuals:remove(i_orig)
    end
end

local function copy_materialize_info(orig)
    local result = A.MaterializeInfo(orig.domains)
    result.materialize,result.sparse = orig.materialize,orig.sparse
    return result
end

function A.NamedResidual:assume_schedule(other)
    assert(A.IterationDomain(self.domains) == A.IterationDomain(other.domains))
    self._domain_order = other._domain_order
    self.J = copy_materialize_info(other.J)
    self.JtJ = copy_materialize_info(other.JtJ)
    self.Jp = copy_materialize_info(other.Jp)
    self.JtF = A.JTFInfo(other.JtF._compute_at_output)
end

function A.NamedResidual:init()
    self.domains = get_index_domains_from_exp_list(self.expressions)
    self.J = A.MaterializeInfo(self.domains)
    self.JtJ = A.MaterializeInfo(self.domains)
    self.Jp = A.MaterializeInfo(self.domains)
    self.JtF = A.JTFInfo(false)
end

function A.JTFInfo:compute_at_output(b)
    self._compute_at_output = b
end

function A.NamedResidual:compute_at_output(b)
    self._compute_at_output = b
    return self
end

function A.MaterializeInfo:init()
    self.materialize = false
    self.sparse = false
end

function A.MaterializeInfo:set_materialize(v)
    self.materialize = v
end

function A.MaterializeInfo:set_sparse(v)
    self.sparse = v
end

function A.ProblemSpecAD:Energy(residuals)
    local named_residuals = List()
    for k,rg in pairs(residuals) do
        local new_R = A.NamedResidual(k, get_flat_residual_list(rg))
        named_residuals:insert(new_R)
    end
    named_residuals:sort(function(a, b) return a.name < b.name end)
    return A.Energy(named_residuals)
end

function A.SampledImage:__call(x,y,c)
    if c or self.op.imagebeingsampled.type.channelcount == 1 then
        assert(not c or c < self.op.imagebeingsampled.type.channelcount, "index out of bounds")
        return self.op(c or 0,x,y)
    else
        local r = {}
        for i = 0,self.op.imagebeingsampled.type.channelcount - 1 do
            r[i+1] = self.op(i,x,y)
        end
        return ad.Vector(unpack(r))
    end
end
local function tosampledimage(im)
    if A.Image:isclassof(im) then
        assert(im:DimCount() == 2, "sampled images must be 2D")
        return ad.sampledimage(im)
    end
    return A.SampledImage:isclassof(im) and im or nil
end
function ad.sampledimage(image,imagedx,imagedy)
    if imagedx then
        imagedx = assert(tosampledimage(imagedx), "expected an image or a sampled image as a derivative")
        imagedy = assert(tosampledimage(imagedy), "expected an image or a sampled image as a derivative")
    end
    local op = ad.newop("sampleimage_"..image.name)
    op.imagebeingsampled = image --not the best place to store this but other ways are more cumbersome
    op.hasconst = true
    function op:generate(exp,args) error("sample image is not implemented directly") end
    function op:getpartials(exp)
        assert(imagedx and imagedy, "image derivatives are not defined for this image and cannot be used in autodiff")
        local x,y = unpack(exp:children())
        return terralib.newlist { imagedx(x,y,exp.const), imagedy(x,y,exp.const) }
    end
    return A.SampledImage(op)
end

local tensor_contraction_count = 1
function A.ProblemSpecAD:TensorContraction(indexdomains,exp)
    indexdomains = List(indexdomains)
    if ad.ExpVector:isclassof(exp) then
        local results = {}
        for i,e in ipairs(exp:expressions()) do
            results[i] = self:TensorContraction(indexdomains,e)
        end
        return ad.Vector(unpack(results))
    end

    exp = assert(ad.toexp(exp),"expected a math expression")
    local unknowns, domains_in_expression = findUnknownsAndDomainsInExpression(exp)

    -- "Typechecking" to make sure the contraction domain is fully contained in the domain of the expression
    local passed_in_domains = {}
    for _,d in ipairs(indexdomains) do
        if not domains_in_expression[d] then
            print("Warning: Domain "..tostring(d).." not found in TensorContraction expression ")
        end
        passed_in_domains[d] = true
    end
    local intermediate_domains = List()
    local contracted_domains = List()
    local all_domains = List()
    for d,_ in pairs(domains_in_expression) do 
        if passed_in_domains[d] then
            contracted_domains:insert(d)
        else
            intermediate_domains:insert(d)
        end
    end

    local function domain_compare(d0,d1) return d0.index < d1.index end
    contracted_domains:sort(domain_compare)
    intermediate_domains:sort(domain_compare)
    all_domains:insertall(contracted_domains)
    all_domains:insertall(intermediate_domains)

    local ispace = toispace(intermediate_domains:map(function(x) return x.dim end))

    local name = "tensor_contraction"..tostring(tensor_contraction_count)
    tensor_contraction_count = tensor_contraction_count + 1

    local im = self:ImageTemporary(name,ispace,1)

    im.indexdomains = intermediate_domains

    im.expression = exp
    local imdom = A.IterationDomain(intermediate_domains)
    local itdom = A.IterationDomain(all_domains)
    im.constraints = constraintsforexpression(imdom,exp)
    self.precomputed:insert(A.PrecomputedDomain(im,itdom,imdom))

    -- flag this access
    im.tensor_contraction = true
    im.contracted_domains = contracted_domains

    im.unknowns = unknowns

    local index = A.ImageIndex(intermediate_domains:map(function(dom)
            return A.DirectIndexComponent(dom)
        end))
    return im(index)
end


function A.SampledImageArray:__call(x,y,z,c)
    if c or self.op.imagebeingsampled.type.channelcount == 1 then
        assert(not c or c < self.op.imagebeingsampled.type.channelcount, "index out of bounds")
        return self.op(c or 0,x,y,z)
    else
        local r = {}
        for i = 0,self.op.imagebeingsampled.type.channelcount - 1 do
            r[i+1] = self.op(i,x,y,z)
        end
        return ad.Vector(unpack(r))
    end
end
local function tosampledimagearray(im)
    if A.Image:isclassof(im) then
        assert(im:DimCount() == 3, "sampled image arrays must be 3D")
        return ad.sampledimagearray(im)
    end
    return A.SampledImageArray:isclassof(im) and im or nil
end
function ad.sampledimagearray(image,imagedx,imagedy)
    if imagedx then
        imagedx = assert(tosampledimagearray(imagedx), "expected an image or a sampled image as a derivative")
        imagedy = assert(tosampledimagearray(imagedy), "expected an image or a sampled image as a derivative")
    end
    local op = ad.newop("sampleimagearray_"..image.name)
    op.imagebeingsampled = image --not the best place to store this but other ways are more cumbersome
    op.hasconst = true
    function op:generate(exp,args) error("sample image array is not implemented directly") end
    function op:getpartials(exp)
        print("TODO: Fix lack of derivatives for sampled image array")
        return terralib.newlist { 0.0, 0.0 }
        --assert(imagedx and imagedy, "image derivatives are not defined for this image ("..image.name..") and cannot be used in autodiff")
        --local x,y = unpack(exp:children())
        --return terralib.newlist { imagedx(x,y,exp.const), imagedy(x,y,exp.const) }
    end
    return A.SampledImageArray(op)
end

for i = 2,12 do
    local s = tostring(i)
    thallo["float"..s] = util.Vector(float,i)
    thallo["double"..s] = util.Vector(double,i)
    if thallo_float == float then
        thallo["thallo_float"..s] = thallo["float"..s]
    else
        thallo["thallo_float"..s] = thallo["double"..s]
    end
end

for i = 2,4 do
    local s = tostring(i)
    thallo["mat"..s.."f"] = util.Vector(float,i*i)
    thallo["mat"..s.."d"] = util.Vector(double,i*i)
    if thallo_float == float then
        thallo["thallo_mat"..s.."f"] = thallo["mat"..s.."f"]
    else
        thallo["thallo_mat"..s.."f"] = thallo["mat"..s.."d"]
    end
end

thallo.Dot = util.Dot
thallo.toispace = toispace

-- C API implementation functions
-- WARNING: if you change these you need to update release/Thallo.h

-- define just stores meta-data right now. ProblemPlan does all compilation for now
terra thallo.ProblemDefine(filename : rawstring, kind : rawstring)
    var id : int
    problemDefine(filename, kind, &id)
    return [&thallo.Problem](id)
end 
terra thallo.ProblemDelete(p : &thallo.Problem)
    var id = int(int64(p))
    problemDelete(id)
end
terra thallo.ProblemPlan(problem : &thallo.Problem, dimensions : &uint32) : &thallo.Plan
    var p : &thallo.Plan = nil 
    problemPlan(int(int64(problem)),dimensions,&p,-1,-1)
    return p
end

terra thallo.PlanFree(plan : &thallo.Plan)
    plan.free(plan.data)
    planFree(plan)
end

terra thallo.ProblemInit(plan : &thallo.Plan, params : &&opaque) : {}
    plan.init(plan.data, params)
end
terra thallo.ProblemStep(plan : &thallo.Plan, params : &&opaque) : int
    return plan.step(plan.data, params)
end
terra thallo.ProblemSolve(plan : &thallo.Plan, params : &&opaque)
   thallo.ProblemInit(plan, params)
   while thallo.ProblemStep(plan, params) ~= 0 do end
end
terra thallo.ProblemCurrentCost(plan : &thallo.Plan) : double
    return plan.cost(plan.data)
end

terra thallo.SetSolverParameter(plan : &thallo.Plan, name : rawstring, value : &opaque) 
    return plan.setsolverparameter(plan.data, name, value)
end

terra thallo.GetSolverParameter(plan : &thallo.Plan, name : rawstring, value : &opaque) 
    return plan.getsolverparameter(plan.data, name, value)
end

terra thallo.GetPerformanceSummary(plan : &thallo.Plan, summary : &Thallo_PerformanceSummary)
    C.memset(summary, 0, sizeof(Thallo_PerformanceSummary))
    plan.get_summary(plan.data,summary)
end

return thallo

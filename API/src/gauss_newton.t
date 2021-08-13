local S = require("std")
local util = require("util")
require("precision")

local cu = nil
if _thallo_use_cpu_only==1 then
    cu = require("cpu_cuda")
else
    cu = require("cuda_util")
end

local Thallo_PerformanceSummary    = util.Thallo_PerformanceSummary
local Thallo_PerformanceEntry      = util.Thallo_PerformanceEntry

-- Simple language extension so we don't have to write
-- escape if [pred] then emit quote 
--      [stmts] 
-- end end end
import "maybe_emit"

local compute_intermediate_cost = false
local enable_direct_solve = false
local use_dense_where_possible = true

local ffi = require("ffi")
local A = util.A
local C = util.C
local Timer = util.Timer

local GuardedInvertType = { CERES = {}, MODIFIED_CERES = {}, EPSILON_ADD = {} }

-- CERES default, ONCE_PER_SOLVE
local JacobiScalingType = { NONE = {}, ONCE_PER_SOLVE = {}, EVERY_ITERATION = {}}


local initialization_parameters = {
    guardedInvertType = GuardedInvertType.CERES,
    jacobiScaling = JacobiScalingType.ONCE_PER_SOLVE
}

local solver_parameter_defaults = {
    residual_reset_period = 10,
    min_relative_decrease = 1e-3,
    min_trust_region_radius = 1e-32,
    max_trust_region_radius = 1e16,
    q_tolerance = 0.0001,
    function_tolerance = 0.000001,
    trust_region_radius = 1e4,
    radius_decrease_factor = 2.0,
    min_lm_diagonal = 1e-6,
    max_lm_diagonal = 1e32,
    max_solver_time_in_seconds = 0, -- 0 Indicates no maximum
    nIterations = 10,
    lIterations = 10
}


local cd = util.cd
local gpuMath = util.gpuMath

local CUBLAS = nil
local CUsp = nil
local cb = nil
local cs = nil


local logDebugCudaThalloFloat
if _thallo_verbosity > 2 then 
    logDebugCudaThalloFloat = macro(function(name,val)
        return quote
            var h_val : thallo_float
            C.cudaMemcpy(&h_val, val, sizeof(thallo_float), C.cudaMemcpyDeviceToHost)
            C.printf("%s: %.20g\n", name, h_val)
        end
    end)
else
    logDebugCudaThalloFloat = macro(function(name,val)
        return 0
    end)
end

local logDebugCudaThalloFloatBuffer = macro(function(name,val,count)
    return quote
        var h_val = [&thallo_float](C.malloc(sizeof(thallo_float)*count))
        C.cudaMemcpy(h_val, val, sizeof(thallo_float)*count, C.cudaMemcpyDeviceToHost)
        C.printf("%s (%d): {\n", name,count)
        for i=0,count do
            C.printf(" %d: %.20g\n", i, h_val[i])
        end
        C.printf("} (end %s)\n", name)
        C.free(h_val)
    end
end)

local logDebugCudaThalloIntBuffer = macro(function(name,val,count)
    return quote
        var h_val = [&int](C.malloc(sizeof(int)*count))
        C.cudaMemcpy(h_val, val, sizeof(int)*count, C.cudaMemcpyDeviceToHost)
        C.printf("%s(%p) (%d): {\n", name,val,count)
        for i=0,count do
            C.printf(" %d: %d\n", i, h_val[i])
        end
        C.printf("} (end %s)\n", name)
        C.free(h_val)
    end
end)
--
logDebugCudaThalloFloatBuffer = macro(function(name,val,count)
    return 0
end)
logDebugCudaThalloIntBuffer = logDebugCudaThalloFloatBuffer
--]]
local FLOAT_EPSILON = `[thallo_float](0.00000001f) 
-- GAUSS NEWTON (or LEVENBERG-MARQUADT)
return function(problemSpec,dimensions)

    local use_direct_solve = enable_direct_solve and (not problemSpec:UsesLambda()) and problemSpec:DirectSolve()
    local use_cublas = problemSpec:RequiresJtJMaterialize()--use_direct_solve
    local use_cusparse = problemSpec:RequiresJ()

    local use_direct_solve = enable_direct_solve and use_cublas and (not problemSpec:UsesLambda()) and problemSpec:DirectSolve()

    local UnknownType = problemSpec:UnknownType()
    local TUnknownType = UnknownType:terratype()

    local ResidualType = problemSpec:ResidualType()
    local TResidualType = ResidualType:terratype()

    local JTJType = problemSpec:JTJType()
    local TJTJType = JTJType:terratype()


    -- start of the unknowns that correspond to this image
    -- for each entry there are a constant number of unknowns
    -- corresponds to the col dim of the J matrix
    local imagename_to_unknown_offset = {}
    
    -- start of the rows of residuals for this energy spec
    -- corresponds to the row dim of the J matrix
    local residualgroups_to_residual_offset_exp = {}
    
    -- start of the block of non-zero entries that correspond to this energy spec
    -- the total dimension here adds up to the number of non-zeros
    local residualgroups_to_rowidx_offset_exp = {}


    local nUnknowns,nResidualsExp,nnzExp = 0,`0,`0
    local nResidualsMaterializedExp,nnzMaterializedExp = `0,`0
    local parametersSym = symbol(&problemSpec:ParameterType(),"parameters")
    local function numberofelements(RG)
        return RG:cardinality()
    end
    local materialized_is_dense = false
    local rgs = problemSpec.scheduledenergy.residualgroups
    assert(rgs)
    
    for i,image in ipairs(UnknownType.images) do
        imagename_to_unknown_offset[image.name] = nUnknowns
        nUnknowns = nUnknowns + image.imagetype:cardinality()
    end
    print(nUnknowns)
    print(UnknownType:cardinality())
    assert(nUnknowns == UnknownType:cardinality(), "Unknown cardinality inconsistent")
    materialized_is_dense = true
    local total_nentries = 0
    local total_possible_nentries = 0
    local materializeJScheds = {A.PRECOMPUTE_J, A.PRECOMPUTE_J_THEN_JTJ}
    for i,rg in ipairs(rgs) do
        residualgroups_to_residual_offset_exp[rg] = nResidualsMaterializedExp
        residualgroups_to_rowidx_offset_exp[rg] = nnzMaterializedExp
        local residuals_per_element = #rg.residuals            
        local nentries = rg:jacobianEntriesPerElement()
        nResidualsExp = `nResidualsExp + [numberofelements(rg)]*residuals_per_element
        nnzExp = `nnzExp + [numberofelements(rg)]*nentries
        local jtjp_sched = rg.domainandschedule.schedule.jtjpschedule
        if util.exists(materializeJScheds, function(t) return t:isclassof(jtjp_sched) end) then
            nResidualsMaterializedExp = `nResidualsMaterializedExp + [numberofelements(rg)]*residuals_per_element
            nnzMaterializedExp = `nnzMaterializedExp + [numberofelements(rg)]*nentries
            total_nentries = total_nentries + nentries
            total_possible_nentries = total_possible_nentries + (residuals_per_element*nUnknowns)
        end
    end
    local density = total_nentries / total_possible_nentries
    materialized_is_dense = (density == 1.0) and use_dense_where_possible
    print("nUnknowns = ",nUnknowns)
    print("nResiduals = ",nResidualsExp)
    print("nnz = ",nnzExp)
    print("nResidualsMaterialized = ",nResidualsMaterializedExp)
    print("nnzMaterialized = ",nnzMaterializedExp)
    print("materialized_is_dense =",materialized_is_dense)


    if (use_cublas or (use_cusparse and materialized_is_dense)) and not CUBLAS then
        CUBLAS,cb = cu.loadCUDALibrary("cublas","cublas_v2.h", "CUBLAS_STATUS_SUCCESS")
    end
    if use_cusparse and not CUsp then
        CUsp,cs = cu.loadCUDALibrary("cusparse","cusparse_v2.h", "CUSPARSE_STATUS_SUCCESS")
    end
    
    local struct SolverParameters {
        min_relative_decrease : float
        min_trust_region_radius : float
        max_trust_region_radius : float
        q_tolerance : float
        function_tolerance : float
        trust_region_radius : float
        radius_decrease_factor : float
        min_lm_diagonal : float
        max_lm_diagonal : float
        max_solver_time_in_seconds : float

        residual_reset_period : int
        nIter : int             --current non-linear iter counter
        nIterations : int       --non-linear iterations
        lIterations : int       --linear iterations
    }

    local struct CSRMatrix {
        data    : &float -- TODO: handle double precision
        colInd  : &int
        rowPtr  : &int
        nnz : int -- Number of nonzeros
        numRows : int --Number of rows
    }

    local safeDivideIfNotLM = macro(function(result,numerator,denominator) return quote
            if [problemSpec:UsesLambda()] then
                result = numerator/denominator
            else
                if denominator ~= [thallo_float](0.0f) then
                    result = numerator/denominator
                end
            end
        end end)

    -- TODO: compute this well
    local multistep_alphaDenominator_compute = not problemSpec:CanFuseJtJpReduction()
    local struct HostData {
        plan : thallo.Plan
        solverparameters : SolverParameters
        timer : Timer
        endSolver : util.TimerEvent

        nonlinearIterationEvent   : util.TimerEvent

        nonlinearSetupEvent   : util.TimerEvent
        linearIterationsEvent : util.TimerEvent
        nonlinearResultsEvent : util.TimerEvent

        queryEvent : C.cudaEvent_t

        prevCost : thallo_float
        perfSummary : Thallo_PerformanceSummary
        finalized : bool
    }
    if CUBLAS then
        HostData.entries:insert {"blas_handle", CUBLAS.cublasHandle_t  }
        --PlanData.entries:insert {"desc", CUBLAS.cusparseMatDescr_t }
        HostData.entries:insert {"cublasJTJEvent", util.TimerEvent  }
        if use_direct_solve then
            HostData.entries:insert {"cublasSolveEvent", util.TimerEvent  }
            HostData.entries:insert {"cublasSolveMatPtr", &&thallo_float  }

            HostData.entries:insert {"cublasSolveVecPtr", &&thallo_float  }
            HostData.entries:insert {"cublasSolveSolutionPtr", &&thallo_float  }
            HostData.entries:insert {"cublasSolveInvMat", &thallo_float  }

            HostData.entries:insert {"cublas_PivotArray", &int  }
            HostData.entries:insert {"cublas_infoArray", &int  }
        end
    end
    if use_cusparse then
        HostData.entries:insert {"cusp_handle",  CUsp.cusparseHandle_t  }
        HostData.entries:insert {"cusp_desc",    CUsp.cusparseMatDescr_t }
        HostData.entries:insert {"cusparse_JT",  CSRMatrix  }
        HostData.entries:insert {"cusparse_JTJ", CSRMatrix  }
        HostData.entries:insert {"cusparse_Jp", &thallo_float  }
        HostData.entries:insert {"cusparse_permutation", &int}
    end
    S.Object(HostData)

    local struct PlanData {
        parameters : problemSpec:ParameterType()

        scratch : &thallo_float

        delta : TUnknownType	--current linear update to be computed -> num vars
        r : TUnknownType		--residuals -> num vars	
        b : TUnknownType        --J^TF. Constant during inner iterations, only used to recompute r to counteract drift -> num vars
        Adelta : TUnknownType       -- (A'A+D'D)delta 
        z : TUnknownType		--preconditioned residuals -> num vars
        p : TUnknownType		--descent direction -> num vars
        Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
        CtC : TUnknownType -- The diagonal matrix C'C for the inner linear solve (J'J+C'C)x = J'F Used only by LM
        preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
        SSq : TUnknownType -- Square of jacobi scaling diagonal

        prevX : TUnknownType -- Place to copy unknowns to before speculatively updating. Avoids hassle when (X + delta) - delta != X 
        initX : TUnknownType

        scanAlphaNumerator : &thallo_float
        scanAlphaDenominator : &thallo_float
        scanBetaNumerator : &thallo_float

        modelCost : &thallo_float    -- modelCost = L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
        q : &thallo_float -- Q value for zeta calculation (see CERES)

        hd : &HostData-- For data that doesn't need to be accessed from kernels
    }
    if use_cublas then
        PlanData.entries:insert {"cublas_jtj", &thallo_float  }
    end
    if use_cusparse then
        PlanData.entries:insert {"cusparse_J",   CSRMatrix  }
        PlanData.entries:insert {"cusparse_unsortedJ",   &thallo_float  }
    end
    if problemSpec:RequiresSeparateJtAndJ() then
        PlanData.entries:insert {"Jp",  TResidualType}
    end
    if problemSpec:RequiresJtJMaterialize() then
        PlanData.entries:insert {"JTJ", TJTJType}
    end
    S.Object(PlanData)

    local generateDumpJ
    if use_cusparse then
        generateDumpJ = function (ES,computeJ,idx,pd)
            local nnz_per_entry = ES:jacobianEntriesPerElement()
            local base_rowidx = residualgroups_to_rowidx_offset_exp[ES]
            local base_residual = residualgroups_to_residual_offset_exp[ES]
            local statements = terralib.newlist()

            local parametersSym = symbol(&problemSpec:ParameterType(),"parameters")
            -- This is duplicated functionality from thallo.t
            -- TODO: dedupe
            local getIndex --forward declaration
            local index_cache = {}
            local function getindirectindexvalue(access, statements)
                local sparse = access.sparse
                local sparseInd = access.index
                local edIndex = getIndex(sparse.inspace:indextype(),sparseInd,statements)
                return `parametersSym.[sparse.name][edIndex].["d0"]
            end
            -- TODO: is full correct?
            local res_domain = ES.domainandschedule.domain.full
            local function get_domain_index(domain)
                local j = table.indexOf(res_domain.domains, domain)
                assert(j>0, "Can't find index domain "..tostring(domain).." in fnkind domain: "..tostring(res_domain))
                return j
            end

            local function get_index_component_value(component, statements)
                local function recurse(c) return get_index_component_value(c, statements) end
                local c = component
                if A.DirectIndexComponent:isclassof(c) then
                    local domain = c.domain
                    local j = get_domain_index(domain)
                    return `idx.["d"..tostring(j-1)]
                elseif A.SparseIndexComponent:isclassof(c) then
                    return getindirectindexvalue(c.access, statements)
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
            getIndex = function (indextype,ind,statements)
                local index = index_cache[ind] 
                if not index then
                    local imIndexType = indextype
                    index = symbol(imIndexType,tostring(ind))
                    --print("Inserting to cache: "..tostring(index))
                    statements:insert( quote var [index] end )
                    for i=1,#ind.components do
                        local value = get_index_component_value(ind.components[i], statements)
                        statements:insert( quote [index].["d"..tostring(i-1)] = value end )
                    end
                end
                index_cache[ind] = index
                return `index:wrap():tooffset()
            end
            local ex_domain = ES.domainandschedule.domain.external

            
            local ridx_offset
            -- Compute residual index instead
            if idx.type == int then
                ridx_offset = idx
            else
                ridx_offset = getIndex(ex_domain:IndexSpace():indextype(),
                    ex_domain:ZeroOffsetIndex(),statements)
            end
            local local_rowidx = `base_rowidx + ridx_offset*nnz_per_entry
            local local_residual = `base_residual + ridx_offset*[#ES.residuals]


            local symJ = symbol(&CSRMatrix,"J")
            local rhs = symbol(computeJ:gettype().returntype, "rhs")
            local scalar_return = (computeJ:gettype().returntype == thallo_float)

            statements:insert(quote
                var [parametersSym] = &pd.parameters 
                var [rhs] = computeJ(idx,pd.parameters)
                var [symJ] = &pd.cusparse_J 
            end)
            local nnz = 0
            local nunknowns = 0
            local residual = 0
            local rg_domain = ES.domainandschedule.domain
            local inner_domains = util.list_subtraction(rg_domain.full.domains, rg_domain.external.domains)

            local jData = symbol(&thallo_float,"jData")
            if materialized_is_dense then
                statements:insert(quote var [jData] = symJ.data end)
            else
                statements:insert(quote var [jData] = pd.cusparse_unsortedJ end)
            end

            local base_thread = symbol(bool,"base_thread")
            statements:insert(quote
                var [base_thread] = true
            end)
            for _,d in ipairs(inner_domains) do
                local j = get_domain_index(d)
                statements:insert(quote
                    [base_thread] = [base_thread] and (idx.["d"..tostring(j-1)] == 0)
                end)
            end

            for _,r in ipairs(ES.residuals) do
                statements:insert(quote
                    if base_thread then
                        symJ.rowPtr[local_residual+residual] = local_rowidx + nnz
                    end
                    --printf("Setting J.rowPtr(%p)[%d] = %d\n", symJ.rowPtr, local_residual+residual, local_rowidx + nnz)
                end)
                local begincolumns = nnz
                -- sort w.r.t. the index *before* concretization?
                for i,u in ipairs(r.unknowns) do
                    local image_offset = imagename_to_unknown_offset[u.image.name]
                    local nchannels = u.image.type.channelcount
                    local uidx = getIndex(u.image:indextype(),u.index,statements)
                    local unknown_index = `image_offset + nchannels*uidx + u.channel

                    local nnz_for_unknown = 1
                    local unknown_domains = util.extract_domains_from_index(u.index)
                    local inner_unknown_domains = terralib.newlist()
                    for _,d in ipairs(unknown_domains) do
                        if table.indexOf(inner_domains,d) > 0 then
                            nnz_for_unknown = nnz_for_unknown*d.dim.size
                            inner_unknown_domains:insert(d)
                        end
                    end
                    local local_unknown_idx = 0
                    if #inner_unknown_domains > 0 then
                        local itDom = A.IterationDomain(inner_unknown_domains)
                        local_unknown_idx = getIndex(itDom:IndexSpace():indextype(),
                            itDom:ZeroOffsetIndex(),statements)
                    end

                    if scalar_return then
                        statements:insert(quote
                            jData[local_rowidx + nnz + local_unknown_idx]   = rhs
                            symJ.colInd[local_rowidx + nnz + local_unknown_idx] = unknown_index
                        end)
                    else
                        statements:insert(quote
                            jData[local_rowidx + nnz + local_unknown_idx]   = thallo_float(rhs.["_"..tostring(nunknowns)])
                            symJ.colInd[local_rowidx + nnz + local_unknown_idx] = unknown_index
                        end)
                    end
                    
                    nnz = nnz + nnz_for_unknown
                    nunknowns = nunknowns + 1
                end
                residual = residual + 1
            end
            return quote [statements] end
        end -- generateDumpJ
    end -- if use_cusparse



    local delegate = {}

    local function numTrianglularElements(n)
        return (n*(n+1))/2
    end

    function delegate.JTJwiseFunctions()
        local kernels = {}
        --TODO: Improve
        if problemSpec:RequiresJtJMaterialize() and use_cublas then
            terra kernels.JTJ_CUBLAS_Setup(pd : PlanData)
                var rawIdx : int = blockDim.x * blockIdx.x + threadIdx.x
                var rowId : int = rawIdx % [nUnknowns]
                var colId : int = rawIdx / [nUnknowns]
                if (rawIdx < [nUnknowns*nUnknowns]) then 
                    var result : float = 0.0f
                    escape
                        local sofar = 0
                        for i,u0 in ipairs(problemSpec:UnknownType().images) do
                            local s0 = u0.imagetype.ispace:cardinality()
                            local maxVal = sofar + u0.imagetype:cardinality()
                            emit quote
                                if rowId >= sofar and rowId < maxVal then
                                    var internal0idx = rowId-sofar
                                    var i0 = internal0idx / u0.imagetype.channelcount
                                    var c0 = internal0idx % u0.imagetype.channelcount
                                    escape
                                        local sofarcol = 0
                                        for j,u1 in ipairs(problemSpec:UnknownType().images) do
                                            local jtjblockIdx = (i-1)*(#problemSpec:UnknownType().images)+j
                                            local jtjblock = JTJType.blocks[jtjblockIdx]
                                            local jtjname = jtjblock.name
                                            local maxValCol = sofarcol + u1.imagetype:cardinality()
                                            emit quote
                                                if colId >= sofarcol and colId < maxValCol then
                                                    var internal1idx = colId-sofarcol
                                                    var i1 = internal1idx / u1.imagetype.channelcount
                                                    var c1 = internal1idx % u1.imagetype.channelcount
                                                    var jtjIndex = (i1*s0+i0)*[jtjblock.imagetype.channelcount]+(c0*[u1.imagetype.channelcount]+c1)
                                                    var jtj_ij = pd.JTJ.[jtjname]:rawScalarGet(jtjIndex)
                                                    result = jtj_ij
                                                end
                                            end
                                            sofarcol = maxValCol
                                        end
                                    end
                                end
                            end
                            sofar = maxVal
                        end
                    end
                    -- The janky LU solve requires an invertible matrix...
                    if [use_direct_solve and not problemSpec:UsesLambda()] and colId == rowId 
                        then result = result + 0.0000001f 
                    end
                    pd.cublas_jtj[rawIdx] = result
                end
            end
        end
        return kernels
    end

    function delegate.FlatUnknownwiseFunctions()
        local kernels = {}
        local Index = int
        local nUnknownTypes = #problemSpec:UnknownType().images

        if problemSpec:RequiresJtJMaterialize() then
            terra kernels.PCGStep1_materializedJTJ(pd : PlanData)
                var row : Index = blockDim.x * blockIdx.x + threadIdx.x
                -- TODO: don't recompute nUnknowns in two places
                if row < [nUnknowns] then
                    var result : float = 0.0f
                    var outputImage : &thallo_float
                    var internal0idx : int
                    var s0 : int
                    var i0 : int
                    var c0 : int
                    var channels0 : int
                    var jtjBlocks : (&thallo_float)[nUnknownTypes][nUnknownTypes]
                    var iJBlock : int
                    escape
                        local sofar = 0
                        for i,u0 in ipairs(problemSpec:UnknownType().images) do
                            local maxVal = sofar + u0.imagetype:cardinality()
                            emit quote
                                if row >= sofar and row < maxVal then
                                    outputImage = [&thallo_float](pd.Ap_X.[u0.name].data)
                                    internal0idx = row-sofar
                                    i0 = internal0idx / u0.imagetype.channelcount
                                    c0 = internal0idx % u0.imagetype.channelcount
                                    s0 = [u0.imagetype.ispace:cardinality()]
                                    channels0 = [u0.imagetype.channelcount]
                                    iJBlock = [i-1]
                                end
                            end
                            
                            for j=1,nUnknownTypes do
                                local jtjblockIdx = ((i-1)*nUnknownTypes)+j
                                local jtjblock = JTJType.blocks[jtjblockIdx]
                                local jtjname = jtjblock.name
                                emit quote 
                                    jtjBlocks[i-1][j-1] = [&thallo_float](pd.JTJ.[jtjname].data)
                                end
                            end
                            
                            sofar = maxVal
                        end
                    end
                    
                    escape
                        local minIter = 0
                        for j,u1 in ipairs(problemSpec:UnknownType().images) do
                            local maxIter = minIter + u1.imagetype:cardinality()
                            emit quote
                                for col=minIter,maxIter do
                                    var internal1idx = col-minIter
                                    var i1 = internal1idx / u1.imagetype.channelcount
                                    var c1 = internal1idx % u1.imagetype.channelcount
                                    var jtjIndex = (i1*s0+i0)*channels0*u1.imagetype.channelcount+(c0*[u1.imagetype.channelcount]+c1)
                                    var jtj_ij = jtjBlocks[iJBlock][j-1][jtjIndex]
                                    var p_j = pd.p.[u1.name]:rawScalarGet(internal1idx)
                                    result = result + (jtj_ij*p_j)
                                end
                            end
                            minIter = maxIter
                        end
                    end
                    outputImage[internal0idx] = result
                end
            end
        end
        return kernels
    end

    local Reduce = util.reduce

    function delegate.UnknownwiseFunctions(UnknownIndexSpace,fmap)
        local kernels = {}
        local unknownElement = UnknownType:VectorTypeForIndexSpace(UnknownIndexSpace)
        local Index = UnknownIndexSpace:indextype()

        local terra square(x : thallo_float) : thallo_float
            return x*x
        end

        local terra guardedInvert(p : unknownElement)
            escape 
                local git = initialization_parameters.guardedInvertType
                if git == GuardedInvertType.CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [thallo_float](1.f) / square(thallo_float(1.f) + util.gpuMath.sqrt(invp(i)))
                        end
                        return invp
                    end
                elseif git == GuardedInvertType.MODIFIED_CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                             invp(i) = [thallo_float](1.f) / (thallo_float(1.f) + invp(i))
                        end
                        return invp
                    end
                elseif git == GuardedInvertType.EPSILON_ADD then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [thallo_float](1.f) / (FLOAT_EPSILON + invp(i))
                        end
                        return invp
                    end
                end
            end
        end

        local terra clamp(x : unknownElement, minVal : unknownElement, maxVal : unknownElement) : unknownElement
            var result = x
            for i = 0, result:size() do
                result(i) = util.gpuMath.fmin(util.gpuMath.fmax(x(i), minVal(i)), maxVal(i))
            end
            return result
        end

        if fmap.evalJTFUnknownwise then
            terra kernels.PCGInit1(pd : PlanData)
                var d : thallo_float = thallo_float(0.0f) -- init for out of bounds lanes
            
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
                    var residuum : unknownElement = 0.0f
                    var pre : unknownElement = 0.0f 
                
                    pd.delta(idx) = thallo_float(0.0f)   
                
                    residuum, pre = fmap.evalJTFUnknownwise(idx, pd.parameters)
                    residuum = -residuum
                    pd.r(idx) = residuum
                    if [problemSpec:CanFusePCGInit()] then
                        if not problemSpec.usepreconditioner then
                            pre = thallo_float(1.0f)
                        end  
                        pre = guardedInvert(pre)
                        pd.preconditioner(idx) = pre
                        var p = pre*residuum    -- apply pre-conditioner M^-1              
                        pd.p(idx) = p
                    
                        d = residuum:dot(p) 
                    else
                        pd.preconditioner(idx) = pre
                    end
                end 
                if [problemSpec:CanFusePCGInit()] and not use_direct_solve then
                    Reduce(pd.scanAlphaNumerator,d)
                end
            end
        end

        terra kernels.PCGInit1_Finish(pd : PlanData)
            var d : thallo_float = thallo_float(0.0f) -- init for out of bounds lanes
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var residuum = pd.r(idx)
                var pre : unknownElement
                if [problemSpec.usepreconditioner] then
                    pre = guardedInvert(pd.preconditioner(idx))
                else
                    pre = thallo_float(1.0f)
                end
                var p = pre*residuum	-- apply pre-conditioner M^-1
                pd.preconditioner(idx) = pre
                pd.p(idx) = p
                d = residuum:dot(p)
            end
            if not use_direct_solve then
                Reduce(pd.scanAlphaNumerator, d)
            end
        end

        if fmap.applyJTJUnknownwise then
            terra kernels.PCGStep1(pd : PlanData)
                var d : thallo_float = thallo_float(0.0f)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    var tmp : unknownElement = 0.0f
                     -- A x p_k  => J^T x J x p_k 
                    tmp = fmap.applyJTJUnknownwise(idx, pd.parameters, pd.p, pd.CtC)
                    if [problemSpec:RequiresJtJMaterialize() or use_cusparse] then
                        pd.Ap_X:atomicAdd(idx,tmp)
                    else
                        pd.Ap_X(idx) = tmp                   -- store for next kernel call
                    end
                    d = pd.p(idx):dot(tmp)           -- x-th term of denominator of alpha
                end
                if not [multistep_alphaDenominator_compute] then
                    Reduce(pd.scanAlphaDenominator,d)
                end
            end
        end
        if problemSpec:UsesLambda() then
            if fmap.applyJTJUnknownwise then
                terra kernels.computeAdelta(pd : PlanData)
                    var idx : Index
                    if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                        var tmp : unknownElement = 0.0f
                         -- A x p_k  => J^T x J x p_k 
                        pd.Adelta:atomicAdd(idx,fmap.applyJTJUnknownwise(idx, pd.parameters, pd.delta, pd.CtC))
                    end
                end
            else
                terra kernels.computeAdelta(pd : PlanData)
                    var idx : Index
                    if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                        var tmp : unknownElement = 0.0f
                         -- A x p_k  => J^T x J x p_k 
                        pd.Adelta:atomicAdd(idx,pd.delta(idx)*pd.CtC(idx))
                    end
                end
            end
        end
        if multistep_alphaDenominator_compute then
            -- The diagonal matrix isn't accounted for until this step.
            if (not fmap.applyJTJUnknownwise) and problemSpec:UsesLambda() then
                terra kernels.PCGStep1_Finish(pd : PlanData)
                    var d : thallo_float = thallo_float(0.0f)
                    var idx : Index
                    if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                        var p = pd.p(idx)
                        var Ap_X = pd.Ap_X(idx) + pd.CtC(idx)*p
                        pd.Ap_X(idx) = Ap_X
                        d = p:dot(Ap_X)
                    end
                    Reduce(pd.scanAlphaDenominator, d)
                end
            else
                terra kernels.PCGStep1_Finish(pd : PlanData)
                    var d : thallo_float = thallo_float(0.0f)
                    var idx : Index
                    if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                        d = pd.p(idx):dot(pd.Ap_X(idx))           -- x-th term of denominator of alpha
                        --printf("%d: %f\n", idx, d)
                    end
                    Reduce(pd.scanAlphaDenominator, d)
                end
            end
        end

        terra kernels.PCGStep2(pd : PlanData)
            var betaNum = thallo_float(0.0f) 
            var q = thallo_float(0.0f) -- Only used if LM
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- sum over block results to compute denominator of alpha
                var alphaDenominator : thallo_float    = pd.scanAlphaDenominator[0]
                var alphaNumerator : thallo_float      = pd.scanAlphaNumerator[0]

                -- update step size alpha
                var alpha = thallo_float(0.0f)
                safeDivideIfNotLM(alpha,alphaNumerator,alphaDenominator)                
    
                var delta = pd.delta(idx)+alpha*pd.p(idx)       -- do a descent step
                pd.delta(idx) = delta

                var r = pd.r(idx)-alpha*pd.Ap_X(idx)				-- update residuum
                pd.r(idx) = r										-- store for next kernel call

                var pre : unknownElement
                if [problemSpec.usepreconditioner] then
                    pre = pd.preconditioner(idx)
                else
                    pre = thallo_float(1.0f)
                end
        
                var z = pre*r										-- apply pre-conditioner M^-1
                pd.z(idx) = z;										-- save for next kernel call

                betaNum = z:dot(r)									-- compute x-th term of the numerator of beta

                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = thallo_float(0.5f)*(delta:dot(r + pd.b(idx))) 
                end
            end
            Reduce(pd.scanBetaNumerator, betaNum)
            if [problemSpec:UsesLambda()] then
                Reduce(pd.q,q)
            end
        end

        terra kernels.PCGStep2_1stHalf(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var alphaDenominator : thallo_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : thallo_float = pd.scanAlphaNumerator[0]
                -- update step size alpha
                var alpha = [thallo_float](0.0f)
                safeDivideIfNotLM(alpha,alphaNumerator,alphaDenominator)
                pd.delta(idx) = pd.delta(idx)+alpha*pd.p(idx)       -- do a descent step
            end
        end

        terra kernels.PCGStep2_2ndHalf(pd : PlanData)
            var betaNum = thallo_float(0.0f) 
            var q = thallo_float(0.0f) 
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- Recompute residual
                var Ax = pd.Adelta(idx)
                var b = pd.b(idx)
                var r = b - Ax
                pd.r(idx) = r

                var pre = pd.preconditioner(idx)
                if not problemSpec.usepreconditioner then
                    pre = thallo_float(1.0f)
                end
                var z = pre*r       -- apply pre-conditioner M^-1
                pd.z(idx) = z;      -- save for next kernel call
                betaNum = z:dot(r)        -- compute x-th term of the numerator of beta
                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = thallo_float(0.5f)*(pd.delta(idx):dot(r + b)) 
                end
            end
            Reduce(pd.scanBetaNumerator, betaNum)
            if [problemSpec:UsesLambda()] then
                Reduce(pd.q,q)
            end
        end


        terra kernels.PCGStep3(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var rDotzNew : thallo_float = pd.scanBetaNumerator[0]	-- get new numerator
                var rDotzOld : thallo_float = pd.scanAlphaNumerator[0]	-- get old denominator

                var beta : thallo_float = thallo_float(0.0f)
                safeDivideIfNotLM(beta,rDotzNew,rDotzOld)
                pd.p(idx) = pd.z(idx)+beta*pd.p(idx)                -- update decent direction
            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
            end
        end	
        
        terra kernels.revertUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.prevX(idx)
            end
        end

        terra kernels.savePreviousUnknowns(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.prevX(idx) = pd.parameters.X(idx)
            end
        end

        terra kernels.copyUnknownwise(pd : PlanData, dst : TUnknownType, src : TUnknownType)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                dst(idx) = src(idx)
            end
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGSaveSSq(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                    pd.SSq(idx) = pd.preconditioner(idx)       
                end 
            end

            terra kernels.PCGFinalizeDiagonal(pd : PlanData)
                var idx : Index
                var d = thallo_float(0.0f)
                var q = thallo_float(0.0f)
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                    var unclampedCtC = pd.CtC(idx)
                    var invS_iiSq : unknownElement = thallo_float(1.0f)
                    if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] then
                        invS_iiSq = thallo_float(1.0f) / pd.SSq(idx)
                    elseif [initialization_parameters.jacobiScaling == JacobiScalingType.EVERY_ITERATION] then 
                        invS_iiSq = thallo_float(1.0f) / pd.preconditioner(idx)
                    end -- else if  [initialization_parameters.jacobiScaling == JacobiScalingType.NONE] then invS_iiSq == 1
                    var clampMultiplier = invS_iiSq / pd.parameters.trust_region_radius
                    var minVal = pd.parameters.min_lm_diagonal * clampMultiplier
                    var maxVal = pd.parameters.max_lm_diagonal * clampMultiplier
                    var CtC = clamp(unclampedCtC, minVal, maxVal)
                    pd.CtC(idx) = CtC
                    
                    -- Calculate true preconditioner, taking into account the diagonal
                    var pre = thallo_float(1.0f) / (CtC+pd.parameters.trust_region_radius*unclampedCtC) 
                    pd.preconditioner(idx) = pre
                    var residuum = pd.r(idx)
                    pd.b(idx) = residuum -- copy over to b
                    var p = pre*residuum    -- apply pre-conditioner M^-1
                    pd.p(idx) = p
                    d = residuum:dot(p)
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = thallo_float(0.5f)*(pd.delta(idx):dot(residuum + residuum)) 
                end    
                Reduce(pd.q,q)
                Reduce(pd.scanAlphaNumerator,d)
            end

        end -- :UsesLambda()
        return kernels
    end

    function delegate.IterationDomainwiseFunctions(fnkind,fmap)
        local Index = fnkind:indextype()
        assert(A.IterationDomainwiseFunction:isclassof(fnkind))
        local kernels = {}
        if fmap.precompute then
            terra kernels.precompute(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.precompute(tIdx,pd.parameters)
                end
            end
        end
        return kernels
    end

    function delegate.ResidualwiseFunctions(fnkind,fmap)
        local Index = fnkind:indextype()
        assert(A.ResidualwiseFunction:isclassof(fnkind) or A.ResidualAndContractionwiseFunction:isclassof(fnkind))
        print(fnkind.fnschedule)
        local kernels = {}

        -- TODO: make a generator for kernels
        if fmap.evalJTF then
            terra kernels.PCGInit1(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.evalJTF(tIdx, pd.parameters, pd.r, pd.preconditioner)
                end
            end
        end
        if fmap.applyJTJ then
            terra kernels.PCGStep1(pd : PlanData)
                var d = thallo_float(0.0f)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.p, pd.Ap_X)
                end
                if not [multistep_alphaDenominator_compute] then
                    Reduce(pd.scanAlphaDenominator,d)
                end
            end
        end

        if fmap.computeJ then
            terra kernels.precomputeJ(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    [generateDumpJ(fmap.derivedfrom,fmap.computeJ,tIdx,pd)]
                end
            end
        end

        if fmap.applyJ then
            terra kernels.PCGStep1_J(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.applyJ(tIdx, pd.parameters, pd.p, pd.Jp)
                end
            end
        end

        if fmap.applyJt then
            terra kernels.PCGStep1_Jt(pd : PlanData)
                var tIdx : Index
                var d = thallo_float(0.0f)
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    d = d + fmap.applyJt(tIdx, pd.parameters, pd.Ap_X, pd.p, pd.Jp)
                end
                if not [multistep_alphaDenominator_compute] then
                    Reduce(pd.scanAlphaDenominator,d)
                end
            end
        end

        if fmap.materializeJTJ then
            terra kernels.precomputeJtJ(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.materializeJTJ(tIdx, pd.parameters, pd.JTJ)
                end
            end
        end

        if fmap.applyJTJ then
            terra kernels.computeAdelta(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.applyJTJ(tIdx, pd.parameters, pd.delta, pd.Adelta)
                end
            end
        end

        if fmap.cost then
            terra kernels.computeCost(pd : PlanData)
                var cost : thallo_float = thallo_float(0.0f)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    cost = fmap.cost(tIdx, pd.parameters)
                    --if ((blockDim.x * blockIdx.x + threadIdx.x) % 32) == 0 then 
                        --printf("Cost %d: %g\n", blockDim.x * blockIdx.x + threadIdx.x, cost)
                    --end
                end 
                Reduce(pd.scratch,cost)
            end
        end

        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC(pd : PlanData)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    fmap.computeCtC(tIdx, pd.parameters, pd.CtC)
                end
            end    
            terra kernels.computeModelCost(pd : PlanData)          
                var cost : thallo_float = thallo_float(0.0f)
                var tIdx : Index
                if tIdx:init(blockDim.x * blockIdx.x + threadIdx.x) then
                    cost = fmap.modelcost(tIdx, pd.parameters, pd.delta)
                end 
                Reduce(pd.modelCost,cost)
            end
        end
        return kernels
    end

    local gpu = util.makeGPUFunctions(problemSpec, dimensions, PlanData, delegate, {"computeCost",
                                                                        "PCGInit1",
                                                                        "PCGStep1",
                                                                        "computeAdelta",
                                                                        "PCGComputeCtC",
                                                                        "computeModelCost",
                                                                        "PCGSaveSSq",
                                                                        "PCGFinalizeDiagonal",
                                                                        "PCGInit1_Finish",
                                                                        "PCGStep1_Finish",
                                                                        "PCGStep2_1stHalf",
                                                                        "PCGStep2_2ndHalf",
                                                                        "PCGStep2",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "savePreviousUnknowns",
                                                                        "revertUpdate",
                                                                        "copyUnknownwise",
                                                                        "PCGStep1_J",
                                                                        "PCGStep1_Jt",
                                                                        "precompute",
                                                                        "precomputeJ",
                                                                        "precomputeJtJ",
                                                                        "PCGStep1_materializedJTJ",
                                                                        "JTJ_CUBLAS_Setup"
                                                                        })


    local terra computeCost(pd : &PlanData) : thallo_float
        logTrace("computeCost\n")
        var f : thallo_float = 0.0
        cd(C.cudaMemsetAsync(pd.scratch, 0, sizeof(thallo_float), nil))
        gpu.computeCost(pd)
        cd(C.cudaMemcpy(&f, pd.scratch, sizeof(thallo_float), C.cudaMemcpyDeviceToHost))
        logTrace("Cost: %g\n", f)
        return f
    end

    local terra computeModelCost(pd : &PlanData) : thallo_float
        C.cudaMemsetAsync(pd.modelCost, 0, sizeof(thallo_float), nil)
        gpu.computeModelCost(pd)
        var f : thallo_float
        C.cudaMemcpy(&f, pd.modelCost, sizeof(thallo_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local terra fetchQ(pd : &PlanData) : thallo_float
        var f : thallo_float
        C.cudaMemcpy(&f, pd.q, sizeof(thallo_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local computeModelCostChange
    
    if problemSpec:UsesLambda() then
        terra computeModelCostChange(pd : &PlanData) : thallo_float
            var cost = pd.hd.prevCost
            var model_cost = computeModelCost(pd)
            logSolver(" cost=%g \n",cost)
            logSolver(" model_cost=%g \n",model_cost)
            var model_cost_change = cost - model_cost
            logSolver(" model_cost_change=%g \n",model_cost_change)
            return model_cost_change
        end
    end

    local terra init(data_ : &opaque, params_ : &&opaque)
        --cd(C.cudaGetLastError())
        logTrace("Solver init\n")
        var pd = [&PlanData](data_)
        var hd = pd.hd
        hd.finalized = false
        hd.timer:init()
        hd.timer:startEvent("Total",nil,&hd.endSolver)
        C.cudaEventCreate(&hd.queryEvent)
        logTrace("About to init parameters\n")
        if [util.verboseTrace] then
            [util.validateParameters(`pd.parameters,problemSpec,params_)]
        end
        [util.initParameters(`pd.parameters,problemSpec,params_,true)]
        logTrace("Inited parameters\n")
        var [parametersSym] = &pd.parameters
        logTrace("Params init\n")
        hd.solverparameters.nIter = 0
        [maybe_emit(problemSpec:UsesLambda(), quote 
          pd.parameters.trust_region_radius       = hd.solverparameters.trust_region_radius
          pd.parameters.radius_decrease_factor    = hd.solverparameters.radius_decrease_factor
          pd.parameters.min_lm_diagonal           = hd.solverparameters.min_lm_diagonal
          pd.parameters.max_lm_diagonal           = hd.solverparameters.max_lm_diagonal
        end)]
        logTrace("precompute\n")
        gpu.precompute(pd)
        logTrace("computeCost\n")
        pd.hd.prevCost = computeCost(pd)
        gpu.copyUnknownwise(pd,pd.initX,pd.parameters.X)
        logTrace("Solver init finish\n")
        C.printf("Initial cost: %g\n",pd.hd.prevCost)
        C.printf("Initial cost x 2: %g\n",2.0*pd.hd.prevCost)
    end

    local terra finalize(pd : &PlanData)
        var hd = pd.hd
        if [not compute_intermediate_cost] then
            pd.hd.prevCost = computeCost(pd)
        end
        logSolver("final cost=%g\n", pd.hd.prevCost)
        cd(C.cudaDeviceSynchronize())
        hd.timer:endEvent(nil,hd.endSolver)
        hd.timer:evaluate(&hd.perfSummary)
        C.printf("hd.perfSummary.total.count = %d\n", hd.perfSummary.total.count)
        hd.timer:cleanup()
        hd.finalized = true
    end



    local cublasJTJMatVec = terra(pd : &PlanData, JTJ : &thallo_float) end
    local cublasJTJMul = terra(hd : &HostData, J: &CSRMatrix, JT: &CSRMatrix, JTJ: &CSRMatrix) end
    local cublasJTJMulSplit = terra(hd : &HostData, J: &CSRMatrix, JT: &CSRMatrix, 
        p : &thallo_float, Jp : &thallo_float, JtJp : &thallo_float, needsSummation : bool) end
    if use_cublas or (use_cusparse and materialized_is_dense) then
        local cublasMatVec = CUBLAS.cublasSgemv_v2
        if thallo_float == double then
            cublasMatVec = CUBLAS.cublasDgemv_v2
        end
        cublasJTJMatVec = terra(pd : &PlanData, JTJ : &thallo_float)
            var hd = pd.hd
            if [_thallo_timing_level] > 1 then
                hd.timer:startEvent("CUBLAS JTJ gemv",nil,&hd.cublasJTJEvent)
            end
            var one : thallo_float = 1.0
            var zero : thallo_float = 0.0
            cb(cublasMatVec(
                hd.blas_handle, CUBLAS.CUBLAS_OP_N,
                [nUnknowns], [nUnknowns], &one,
                JTJ, [nUnknowns],
                [&thallo_float](pd.p._contiguousallocation), 1,
                &zero, [&thallo_float](pd.Ap_X._contiguousallocation), 1))
            if [_thallo_timing_level] > 1 then
                hd.timer:endEvent(nil,hd.cublasJTJEvent)
            end
        end
        
        cublasJTJMul = terra (hd : &HostData, J: &CSRMatrix, JT: &CSRMatrix, JTJ: &CSRMatrix)
            var one : thallo_float = 1.0
            var zero : thallo_float = 0.0
            cb(CUBLAS.cublasSgemm_v2(hd.blas_handle, CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_T, JT.numRows, JT.numRows, J.numRows, &one,
                    J.data, JT.numRows, J.data, JT.numRows, &zero,
                    JTJ.data, JT.numRows))
        end
        
        cublasJTJMulSplit = terra(hd : &HostData, J: &CSRMatrix, JT: &CSRMatrix, p: &thallo_float, Jp : &thallo_float, JtJp : &thallo_float, needsSummation : bool) 
            if [_thallo_timing_level] > 1 then
                hd.timer:startEvent("CUBLAS JTJ split gemv",nil,&hd.cublasJTJEvent)
            end
            var one : thallo_float = 1.0
            var zero : thallo_float = 0.0
            cb(cublasMatVec(
               hd.blas_handle, CUBLAS.CUBLAS_OP_T,
               JT.numRows, J.numRows, &one,
               J.data, JT.numRows,
               p, 1,
               &zero, Jp, 1))
            logDebugCudaThalloFloatBuffer("hd.cusparse_Jp", Jp, J.numRows)
            var beta = &zero
            if needsSummation then
                beta = &one
            end
            cb(cublasMatVec(
               hd.blas_handle, CUBLAS.CUBLAS_OP_N,
               JT.numRows, J.numRows, &one,
               J.data, JT.numRows,
               Jp, 1,
               beta, JtJp, 1))
            if [_thallo_timing_level] > 1 then
                hd.timer:endEvent(nil,hd.cublasJTJEvent)
            end
        end
    end

    local cublasDirectSolve = terra(pd : &PlanData) end
    if use_direct_solve then
        local ludecomp = CUBLAS.cublasSgetrfBatched
        local invertMat = CUBLAS.cublasSgetriBatched
        local cublasMatVec = CUBLAS.cublasSgemv_v2
        if thallo_float == double then
            ludecomp = CUBLAS.cublasDgetrfBatched
            invertMat = CUBLAS.cublasDgetriBatched
            cublasMatVec = CUBLAS.cublasDgemv_v2
        end
        cublasDirectSolve = terra(pd : &PlanData)
            var hd = pd.hd
            if [_thallo_timing_level] > 1 then
                hd.timer:startEvent("CUBLAS DirectSolve",nil,&hd.cublasSolveEvent)
            end
            logTrace("About to LU decomp.\n")
            -- step 1: perform in-place LU decomposition, P*A = L*U. 
            -- Aarray[i] is n*n matrix A[i] 
            -- cublasSolveMatPtr
            cb(ludecomp(hd.blas_handle, [nUnknowns], hd.cublasSolveMatPtr, [nUnknowns], 
                hd.cublas_PivotArray, hd.cublas_infoArray, 1))
                    -- check infoArray[i] to see if factorization of A[i] is successful or not. 
            var info : int
            if [util.verboseTrace] then
                cd(C.cudaMemcpy(&info, hd.cublas_infoArray, sizeof(int), C.cudaMemcpyDeviceToHost))
                logTrace("About to invert: %d\n", info)
            end
            -- Array[i] contains LU factorization of A[i] 
            -- step 2: perform out-of-place inversion, Carray[i] = inv(A[i]) 
            cb(invertMat(hd.blas_handle, [nUnknowns], hd.cublasSolveMatPtr, [nUnknowns], 
               hd.cublas_PivotArray, hd.cublasSolveSolutionPtr, [nUnknowns], hd.cublas_infoArray, 1))
            -- check infoArray[i] to see if inversion of A[i] is successful or not. 
            if [util.verboseTrace] then
                cd(C.cudaMemcpy(&info, hd.cublas_infoArray, sizeof(int), C.cudaMemcpyDeviceToHost))
                logTrace("About to matmul: %d\n", info)
            end
            var one : thallo_float = 1.0
            var zero : thallo_float = 0.0
            cb(cublasMatVec(
               hd.blas_handle, CUBLAS.CUBLAS_OP_N,
               [nUnknowns], [nUnknowns], &one,
               hd.cublasSolveInvMat, [nUnknowns],
               [&thallo_float](pd.r._contiguousallocation), 1,
               &zero, [&thallo_float](pd.delta._contiguousallocation), 1))
            if [_thallo_timing_level] > 1 then
                hd.timer:endEvent(nil,hd.cublasSolveEvent)
            end
        end
    end
    local cusparseJTJMatVec = terra(pd : &PlanData, needsSummation : bool) end
    local cusparseOuter = terra(pd : &PlanData) end
    if use_cusparse then
        terra cusparseOuter(pd : &PlanData)
            --logSolver("cusparseOuter start\n")
            var hd : &HostData = pd.hd
            --logSolver("precomputeJ start\n")
            gpu.precomputeJ(pd)
            --logSolver("precomputeJ end\n")

            var J = &pd.cusparse_J
            var JT = &hd.cusparse_JT
            var JTJ = &hd.cusparse_JTJ
            if not materialized_is_dense then
                --logSolver("sort CSR start\n")


                var pBufferSizeInBytes : C.size_t = 0
                var pBuffer : &opaque = nil
                -- step 1: allocate buffer
                cs(CUsp.cusparseXcsrsort_bufferSizeExt(hd.cusp_handle, J.numRows, 
                    nUnknowns, J.nnz, J.rowPtr,J.colInd, &pBufferSizeInBytes))
                
                cd(C.cudaMalloc( &pBuffer, sizeof(uint8)* pBufferSizeInBytes))


                -- step 2: setup permutation vector P to identity
                cs(CUsp.cusparseCreateIdentityPermutation(hd.cusp_handle, J.nnz, hd.cusparse_permutation))

                -- step 3: sort CSR format
                cs(CUsp.cusparseXcsrsort(hd.cusp_handle, J.numRows, 
                    nUnknowns, J.nnz, hd.cusp_desc,J.rowPtr,J.colInd, hd.cusparse_permutation, pBuffer))

                -- step 4: gather sorted csrVal
                cs(CUsp.cusparseSgthr(hd.cusp_handle, J.nnz, pd.cusparse_unsortedJ, J.data, 
                    hd.cusparse_permutation, CUsp.CUSPARSE_INDEX_BASE_ZERO))

                cd(C.cudaFree(pBuffer))
                --logSolver("sort CSR end")

                do
                    --logSolver("J_transpose start\n")
                    var endJtranspose : util.TimerEvent
                    if [_thallo_timing_level] > 1 then
                        hd.timer:startEvent("J_transpose",nil,&endJtranspose)
                    end
                    cs(CUsp.cusparseScsr2csc(hd.cusp_handle,J.numRows, nUnknowns, J.nnz,
                                         J.data,J.rowPtr,J.colInd,
                                         JT.data,JT.colInd,JT.rowPtr,
                                         CUsp.CUSPARSE_ACTION_NUMERIC,CUsp.CUSPARSE_INDEX_BASE_ZERO))
                    if [_thallo_timing_level] > 1 then
                        hd.timer:endEvent(nil,endJtranspose)
                    end
                    --logSolver("J_transpose end\n")
                end
            end
            logDebugCudaThalloIntBuffer("J.rowPtr", J.rowPtr, J.numRows+1)
            logDebugCudaThalloIntBuffer("J.colInd", J.colInd, J.nnz)
            logDebugCudaThalloFloatBuffer("J.data", J.data, J.nnz)


            --logDebugCudaThalloIntBuffer("JT.rowPtr", JT.rowPtr, nUnknowns+1)
            --logDebugCudaThalloIntBuffer("JT.colInd", JT.colInd, JT.nnz)
            --logDebugCudaThalloFloatBuffer("JT.data", JT.data, JT.nnz)
            
            if [problemSpec:RequiresMatMul()] then
                if JTJ.rowPtr == nil then -- Allocate JTJ
                    if materialized_is_dense then
                        cd(C.cudaMalloc([&&opaque](&JTJ.data), sizeof(thallo_float)*JT.numRows*JT.numRows))
                    else
                        --logSolver("Allocate JTJ start\n")
                        --TODO: allow resizing dimensions
                        JTJ.numRows = nUnknowns
                        cd(C.cudaMalloc([&&opaque](&JTJ.rowPtr),sizeof(int)*(JTJ.numRows+1)))
                        var endJTJalloc : util.TimerEvent
                        if [_thallo_timing_level] > 1 then
                            hd.timer:startEvent("J^TJ alloc",nil,&endJTJalloc)
                        end
                        
                        cs(CUsp.cusparseXcsrgemmNnz(hd.cusp_handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              JTJ.numRows,JT.numRows,J.numRows,
                                              hd.cusp_desc,JT.nnz,JT.rowPtr,JT.colInd,
                                              hd.cusp_desc,J.nnz,J.rowPtr,J.colInd,
                                              hd.cusp_desc,JTJ.rowPtr, &JTJ.nnz))
                        if [_thallo_timing_level] > 1 then
                            hd.timer:endEvent(nil,endJTJalloc)
                        end
                        
                        cd(C.cudaMalloc([&&opaque](&JTJ.colInd), sizeof(int)*JTJ.nnz))
                        cd(C.cudaMalloc([&&opaque](&JTJ.data), sizeof(float)*JTJ.nnz))

                        --logSolver("Allocate JTJ end\n")
                    end
                end
                --logSolver("JTJ multiply start\n")
                var endJTJmm : util.TimerEvent
                if [_thallo_timing_level] > 1 then
                    hd.timer:startEvent("JTJ multiply",nil,&endJTJmm)
                end
                if materialized_is_dense then
                    cublasJTJMul(hd,J,JT,JTJ)
                else
                    cs(CUsp.cusparseScsrgemm(hd.cusp_handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            nUnknowns,nUnknowns,J.numRows,
                                              hd.cusp_desc,JT.nnz,JT.data,JT.rowPtr,JT.colInd,
                                              hd.cusp_desc,J.nnz,J.data,J.rowPtr,J.colInd,
                                              hd.cusp_desc,JTJ.data, JTJ.rowPtr,JTJ.colInd ))
                end
                if [_thallo_timing_level] > 1 then
                    hd.timer:endEvent(nil,endJTJmm)
                end
                --logSolver("JTJ multiply end\n")
            end
            --logSolver("cusparseOuter end\n")
            --logDebugCudaThalloIntBuffer("JTJ.rowPtr", JTJ.rowPtr, nUnknowns+1)
            --logDebugCudaThalloIntBuffer("JTJ.colInd", JTJ.colInd, JTJ.nnz)
            --logDebugCudaThalloFloatBuffer("JTJ.data", JTJ.data, JTJ.nnz)
        end

        terra cusparseJTJMatVec(pd : &PlanData, needsSummation : bool)
            --logSolver("cusparseJTJMatVec start\n")
            var hd : &HostData = pd.hd
            var consts = array(0.f,1.f)
            var betaIndex = 0
            if needsSummation then
                betaIndex = 1
                logSolver("Needs Summation!\n")
            end
            cd(C.cudaMemset(pd.Ap_X._contiguousallocation, 0, sizeof(thallo_float)*nUnknowns))
            var J = &pd.cusparse_J
            var JT = &hd.cusparse_JT
            var JTJ = &hd.cusparse_JTJ
            -- TODO: allow different residuals to use different paths
            if [problemSpec:RequiresMatMul()] then
                var endJTJp : util.TimerEvent
                if [_thallo_timing_level] > 1 then
                    hd.timer:startEvent("J^TJp",nil,&endJTJp)
                end
                if materialized_is_dense then
                    cublasJTJMatVec(pd,JTJ.data)
                else
                    cs(CUsp.cusparseScsrmv(
                            hd.cusp_handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nUnknowns, nUnknowns, JTJ.nnz,
                            &consts[1], 
                            hd.cusp_desc,JTJ.data, JTJ.rowPtr,JTJ.colInd,
                            [&thallo_float](pd.p._contiguousallocation),
                            &consts[betaIndex], [&thallo_float](pd.Ap_X._contiguousallocation)
                        ))
                end
                if [_thallo_timing_level] > 1 then
                    hd.timer:endEvent(nil,endJTJp)
                end
            else
                cd(C.cudaMemset(hd.cusparse_Jp,0,sizeof(thallo_float)*J.numRows))

                if materialized_is_dense then
                    var hd = pd.hd
                    cublasJTJMulSplit(hd, J, JT, [&thallo_float](pd.p._contiguousallocation), hd.cusparse_Jp,[&thallo_float](pd.Ap_X._contiguousallocation), needsSummation)
                else
                    var endJp : util.TimerEvent
                    if [_thallo_timing_level] > 1 then
                        hd.timer:startEvent("Jp",nil,&endJp)
                    end
                    cs(CUsp.cusparseScsrmv(
                                hd.cusp_handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                J.numRows, nUnknowns,J.nnz,
                                &consts[1], 
                                hd.cusp_desc,J.data, J.rowPtr,J.colInd,
                                [&float](pd.p._contiguousallocation),
                                &consts[0], hd.cusparse_Jp
                            ))
                    if [_thallo_timing_level] > 1 then
                        hd.timer:endEvent(nil,endJp)
                    end
                    logDebugCudaThalloFloatBuffer("hd.cusparse_Jp", hd.cusparse_Jp, J.numRows)

                    var endJT : util.TimerEvent
                    if [_thallo_timing_level] > 1 then
                        hd.timer:startEvent("J^T",nil,&endJT)
                    end
                    cs(CUsp.cusparseScsrmv(
                                hd.cusp_handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                nUnknowns, J.numRows, J.nnz,
                                &consts[1], 
                                hd.cusp_desc,JT.data, JT.rowPtr,JT.colInd,
                                hd.cusparse_Jp,
                                &consts[betaIndex],[&thallo_float](pd.Ap_X._contiguousallocation) 
                            ))
                    if [_thallo_timing_level] > 1 then
                        hd.timer:endEvent(nil,endJT)
                    end
                end
            end
            logDebugCudaThalloFloatBuffer("pd.Ap_X", [&float](pd.Ap_X._contiguousallocation), nUnknowns)
            --logSolver("cusparseJTJMatVec end\n")
        end
    end
    -- TODO: use contiguous allocation to memset more at once
    local terra clearPerLinearStep(pd : &PlanData)
        var endEvent : C.cudaEvent_t 
        if [_thallo_timing_level] > 1 then
            pd.hd.timer:startEvent("clearPerLinearStep",nil,&endEvent)
        end
        C.cudaMemsetAsync(pd.scanAlphaDenominator, 0, sizeof(thallo_float), nil)
        C.cudaMemsetAsync(pd.scanBetaNumerator, 0, sizeof(thallo_float), nil)
        if [problemSpec:UsesLambda()] then
            C.cudaMemsetAsync(pd.q, 0, sizeof(thallo_float), nil)
        end
        if [_thallo_timing_level] > 1 then
            pd.hd.timer:endEvent(nil,endEvent)
        end
    end

    local do_clears = not (problemSpec:CanFuseJtJpReduction() and problemSpec:CanFusePCGInit())

    local terra step(data_ : &opaque, params_ : &&opaque)
        logTrace("Solver step\n")
        var pd = [&PlanData](data_)
        var hd : &HostData = pd.hd
        var residual_reset_period : int            = hd.solverparameters.residual_reset_period
        var min_relative_decrease : thallo_float      = hd.solverparameters.min_relative_decrease
        var min_trust_region_radius : thallo_float    = hd.solverparameters.min_trust_region_radius
        var max_trust_region_radius : thallo_float    = hd.solverparameters.max_trust_region_radius
        var q_tolerance : thallo_float                = hd.solverparameters.q_tolerance
        var function_tolerance : thallo_float         = hd.solverparameters.function_tolerance
        var max_solver_time_in_seconds : thallo_float = hd.solverparameters.max_solver_time_in_seconds
        var Q0 : thallo_float
        var Q1 : thallo_float
        cd(C.cudaGetLastError())
        [util.initParameters(`pd.parameters,problemSpec, params_,false)]
        cd(C.cudaGetLastError())
        logTrace("Solver step %d/%d\n",hd.solverparameters.nIter,hd.solverparameters.nIterations)
        if hd.solverparameters.nIter < hd.solverparameters.nIterations then
            pd.hd.timer:startEvent("Nonlinear Iteration",nil,&pd.hd.nonlinearIterationEvent)
            pd.hd.timer:startEvent("Nonlinear Setup",nil,&pd.hd.nonlinearSetupEvent)
            if not use_direct_solve then
                cd(C.cudaMemsetAsync(pd.scanAlphaNumerator, 0, sizeof(thallo_float), nil))	--scan in PCGInit1 requires reset
                cd(C.cudaMemsetAsync(pd.scanAlphaDenominator, 0, sizeof(thallo_float), nil))	--scan in PCGInit1 requires reset
                cd(C.cudaMemsetAsync(pd.scanBetaNumerator, 0, sizeof(thallo_float), nil))	--scan in PCGInit1 requires reset
                if [do_clears] then
                    pd.delta:clear()
                    pd.Ap_X:clear()
                end
            end
            if [do_clears] then
                pd.r:clear()
            end

            logTrace("PCGInit\n")
            
            gpu.PCGInit1(pd)
            if not [problemSpec:CanFusePCGInit()] then
                gpu.PCGInit1_Finish(pd)
            end
            if  [problemSpec:RequiresJ()] then
                cusparseOuter(pd)
            end
            [maybe_emit(problemSpec:RequiresJtJMaterialize(), quote
                pd.JTJ:clear()
                gpu.precomputeJtJ(pd)
                if [use_cublas] then
                    gpu.JTJ_CUBLAS_Setup(pd)
                end
            end)]
            logDebugCudaThalloFloat("scanAlphaNumerator", pd.scanAlphaNumerator)
            [maybe_emit(problemSpec:UsesLambda(), quote
                C.cudaMemsetAsync(pd.scanAlphaNumerator, 0, sizeof(thallo_float), nil)
                C.cudaMemsetAsync(pd.q, 0, sizeof(thallo_float), nil)
                if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] and hd.solverparameters.nIter == 0 then
                    gpu.PCGSaveSSq(pd)
                end
                pd.CtC:clear() 
                gpu.PCGComputeCtC(pd)
                -- This also computes Q
                gpu.PCGFinalizeDiagonal(pd)
                Q0 = fetchQ(pd)
            end)]
            
            pd.hd.timer:endEvent(nil,pd.hd.nonlinearSetupEvent)
            --logSolver("Linear Solve\n")
            logTrace("Linear Solve\n")
            pd.hd.timer:startEvent("Linear Solve",nil,&pd.hd.linearIterationsEvent)
            if use_direct_solve then
                cublasDirectSolve(pd)
            else
                for lIter = 0, hd.solverparameters.lIterations do
                    --logSolver("Linear iteration %d/%d\n",lIter,hd.solverparameters.lIterations)
                    logTrace("Linear iteration %d/%d\n",lIter,hd.solverparameters.lIterations)
                    clearPerLinearStep(pd)
                    --logSolver("clearPerLinearStep(pd)\n")
                    
                    var needsSummation = false
                    -- JTJ matvec multiplies do not use atomics, so don't need 
                    -- to clear the output vector beforehand
                    escape
                        if problemSpec:RequiresJtJMaterialize() then
                            if use_cublas then
                                emit `cublasJTJMatVec(pd,pd.cublas_jtj)
                            else
                                emit `gpu.PCGStep1_materializedJTJ(pd)
                            end
                            emit quote needsSummation = true end
                        else
                            if do_clears then
                                emit `pd.Ap_X:clear()
                            end
                        end
                        if use_cusparse then
                            emit `cusparseJTJMatVec(pd,needsSummation)
                        end
                    end
                    if [problemSpec:RequiresApplyJtJp()] then
                        gpu.PCGStep1(pd)
                    end
                    [maybe_emit(problemSpec:RequiresSeparateJtAndJ(), quote
                        pd.Jp:clear()
                        gpu.PCGStep1_J(pd)
                        gpu.PCGStep1_Jt(pd)
                    end)]
                    if multistep_alphaDenominator_compute then
                        gpu.PCGStep1_Finish(pd)
                    end
                    logDebugCudaThalloFloat("scanAlphaDenominator", pd.scanAlphaDenominator)
                    if [problemSpec:UsesLambda()] and ((lIter + 1) % residual_reset_period) == 0 then
                        gpu.PCGStep2_1stHalf(pd)
                        pd.Adelta:clear()
                        gpu.computeAdelta(pd)
                        gpu.PCGStep2_2ndHalf(pd)
                    else
                        gpu.PCGStep2(pd)
                    end
                    gpu.PCGStep3(pd)
                    logDebugCudaThalloFloat("scanBetaNumerator", pd.scanBetaNumerator)

                    -- save new rDotz for next iteration
                    C.cudaMemcpyAsync(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(thallo_float), C.cudaMemcpyDeviceToDevice, nil)
                    if [problemSpec:UsesLambda()] then
                        Q1 = fetchQ(pd)
                        if C.isfinite(Q1) == 0 then
                            logSolver("non-finite Q at iteration: %d\n", lIter+1)
                            break
                        end
                        var zeta = [thallo_float](lIter+1)*(Q1 - Q0) / Q1 
                        if C.isfinite(zeta) == 0 then
                            logSolver("non-finite zeta at iteration: %d\n", lIter+1)
                            break
                        end
                        --logSolver("%d: Q0(%g) Q1(%g), zeta(%g)\n", lIter, Q0, Q1, zeta)
                        if ((lIter+1) % 1000) == 0 then
                            logSolver("zeta=%.18g at iteration: %d\n", zeta, (lIter+1))
                        end
                        if zeta < q_tolerance then
                            logSolver("zeta=%.18g < %.18g, breaking at iteration: %d\n", zeta, q_tolerance, (lIter+1))
                            break
                        end
                        Q0 = Q1
                    end
                end
            end
            pd.hd.timer:endEvent(nil,pd.hd.linearIterationsEvent)
            pd.hd.timer:startEvent("Nonlinear Finish",nil,&pd.hd.nonlinearResultsEvent)

            var model_cost_change : thallo_float

            [maybe_emit(problemSpec:UsesLambda(), quote
                model_cost_change = computeModelCostChange(pd)
                gpu.savePreviousUnknowns(pd)
            end)]
            C.cudaEventRecord(pd.hd.queryEvent, nil)
            gpu.PCGLinearUpdate(pd)

            gpu.precompute(pd)
            var newCost : thallo_float
            if [problemSpec:UsesLambda() or compute_intermediate_cost] then
                newCost = computeCost(pd)
                logSolver("new cost=%g, old cost=%g\n", newCost, pd.hd.prevCost)
            end
            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        var cost_change = pd.hd.prevCost - newCost
                        logSolver(" cost_change=%g \n", cost_change)
                        
                        -- See CERES's TrustRegionStepEvaluator::StepAccepted() for a more complicated version of this
                        var relative_decrease = cost_change / model_cost_change
                        if cost_change >= 0 and relative_decrease > min_relative_decrease then
                            var absolute_function_tolerance = pd.hd.prevCost * function_tolerance
                            if cost_change <= absolute_function_tolerance then
                                logSolver("\nFunction tolerance reached (%g < %g), exiting\n", cost_change, absolute_function_tolerance)
                                pd.hd.timer:endEvent(nil,pd.hd.nonlinearResultsEvent)
                                pd.hd.timer:endEvent(nil,pd.hd.nonlinearIterationEvent)
                                finalize(pd)
                                return 0
                            end

                            var step_quality = relative_decrease
                            var min_factor = 1.0/3.0
                            var tmp_factor = 1.0 - util.cpuMath.pow(2.0 * step_quality - 1.0, 3.0)
                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / util.cpuMath.fmax(min_factor, tmp_factor)
                            pd.parameters.trust_region_radius = util.cpuMath.fmin(pd.parameters.trust_region_radius, max_trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0
                            logSolver(" trust_region_radius=%g \n", pd.parameters.trust_region_radius)
                            pd.hd.prevCost = newCost
                        else 
                            gpu.revertUpdate(pd)

                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / pd.parameters.radius_decrease_factor
                            logSolver(" trust_region_radius=%g \n", pd.parameters.trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0 * pd.parameters.radius_decrease_factor
                            if pd.parameters.trust_region_radius < min_trust_region_radius then
                                logSolver("\nTrust_region_radius is less than the min (%g < %g), exiting\n", pd.parameters.trust_region_radius, min_trust_region_radius)
                                hd.solverparameters.trust_region_radius = 10e4--min_trust_region_radius
                                pd.hd.timer:endEvent(nil,pd.hd.nonlinearResultsEvent)
                                pd.hd.timer:endEvent(nil,pd.hd.nonlinearIterationEvent)
                                finalize(pd)
                                return 0
                            end
                            logSolver("REVERT\n")
                            gpu.precompute(pd)
                        end
                        --TODO: Generalize this...
                        hd.solverparameters.trust_region_radius = pd.parameters.trust_region_radius

                    end
                else
                    if compute_intermediate_cost then
                        emit quote
                            pd.hd.prevCost = newCost 
                        end
                    end
                end 
            end

            pd.hd.solverparameters.nIter = pd.hd.solverparameters.nIter + 1
            pd.hd.timer:endEvent(nil,pd.hd.nonlinearResultsEvent)
            pd.hd.timer:endEvent(nil,pd.hd.nonlinearIterationEvent)

            if max_solver_time_in_seconds > 0.0 then
                
                var solverElapsed : float = 0.0
                -- Check if this stalls us; TODO: don't rely on the first timer event being beginning of solve
                C.cudaEventElapsedTime(&solverElapsed, pd.hd.timer.timingInfo:get(0).startEvent, pd.hd.queryEvent)
                if solverElapsed/1000.0 > max_solver_time_in_seconds then
                    logSolver("\nTime exceeded max (%g > %g), exiting\n", solverElapsed/1000.0, max_solver_time_in_seconds)
                    finalize(pd)
                    return 0
                else
                    logSolver("time elapsed %g\n", solverElapsed/1000.0)
                end
            end
            return 1
        else
            finalize(pd)
            return 0
        end
    end

    local terra cost(data_ : &opaque) : double
        var pd = [&PlanData](data_)
        if (not pd.hd.finalized) and [not compute_intermediate_cost] then
            pd.hd.prevCost = computeCost(pd)
        end
        return [double](pd.hd.prevCost)
    end

    local terra get_summary(data_ : &opaque, summary : &Thallo_PerformanceSummary)
        var pd = [&PlanData](data_)
        C.printf("pd.hd.perfSummary.total.count = %d\n", pd.hd.perfSummary.total.count)
        C.memcpy(summary, &pd.hd.perfSummary, sizeof(Thallo_PerformanceSummary))
    end

    local terra reset_unknowns(data_ : &opaque)
        var pd = [&PlanData](data_)
        gpu.copyUnknownwise(pd,pd.parameters.X,pd.initX)
    end

    local terra initializeSolverParameters(params : &SolverParameters)
        escape
            -- for each value in solver_parameter_defaults, assign to params
            for name,value in pairs(solver_parameter_defaults) do
                local foundVal = false
                -- TODO, more elegant solution to this
                for _,entry in ipairs(SolverParameters.entries) do
                    if entry.field == name then
                        foundVal = true
                        emit quote params.[name] = [entry.type]([value])
                            logTrace(["Initialize Parameter: "..name.."\n"])
                        end
                        break
                    end
                end
                if not foundVal then
                    print("Tried to initialize "..name.." but not found")
                end
            end
        end
    end

    local terra setSolverParameter(data_ : &opaque, name : rawstring, value : &opaque) 
        var pd = [&PlanData](data_)
        escape
            -- Instead of building a table datastructure, 
            -- explicitly emit an if-statement chain for setting the parameter
            for _,entry in ipairs(SolverParameters.entries) do
                emit quote
                    if C.strcmp([entry.field],name)==0 then
                        pd.hd.solverparameters.[entry.field] = @[&entry.type]([value])
                        logTrace("setSolverParameter: %s\n", name)
                        return
                    end
                end
            end
        end
        logSolver("Warning: tried to set nonexistent solver parameter %s\n", name)
    end
    local terra getSolverParameter(data_ : &opaque, name : rawstring, value : &opaque) 
        var pd = [&PlanData](data_)
        escape
            -- Instead of building a table datastructure, 
            -- explicitly emit an if-statement chain for getting the parameter
            for _,entry in ipairs(SolverParameters.entries) do
                emit quote
                    if C.strcmp([entry.field],name)==0 then
                        @[&entry.type]([value]) = pd.hd.solverparameters.[entry.field]
			--                        printf("Setting %s to %g\n", name, pd.hd.solverparameters.[entry.field])
                        logTrace("getSolverParameter: %s\n", name)
                        return
                    end
                end
            end
        end
        logSolver("Warning: tried to set nonexistent solver parameter %s\n", name)
    end

    local terra initCSRMatrix(M : &CSRMatrix, nnz : int, numrows : int)
        cd(C.cudaMalloc([&&opaque](&(M.data)), sizeof(thallo_float)*nnz))
        cd(C.cudaMalloc([&&opaque](&(M.colInd)), sizeof(int)*nnz))
        cd(C.cudaMalloc([&&opaque](&(M.rowPtr)), sizeof(int)*(numrows+1)))
        cd(C.cudaMemset(M.data,0,sizeof(thallo_float)*nnz))
        cd(C.cudaMemset(M.colInd,-1,sizeof(int)*nnz))
        cd(C.cudaMemset(M.rowPtr,-1,sizeof(int)*(numrows+1)))
        M.nnz = nnz
        M.numRows = numrows
        --[[
        logDebugCudaThalloIntBuffer("initCSRMatrix.rowPtr", M.rowPtr, numrows+1)
        logDebugCudaThalloIntBuffer("initCSRMatrix.colInd", M.colInd, M.nnz)
        logDebugCudaThalloFloatBuffer("initCSRMatrix.data", M.data, M.nnz)
    --]]

    end

    local terra estimated_cost(data_ : &opaque) : double
        return [problemSpec.estimated_cost]
    end 

    local terra freeCSRMatrix(M : &CSRMatrix)
        cd(C.cudaFree([&opaque](M.data)))
        cd(C.cudaFree([&opaque](M.colInd)))
        cd(C.cudaFree([&opaque](M.rowPtr)))
    end

    local terra free(data_ : &opaque)
        logTrace("freeing plan\n")
        var pd = [&PlanData](data_)
        pd.delta:freeData()
        pd.r:freeData()
        pd.b:freeData()
        pd.Adelta:freeData()
        pd.z:freeData()
        pd.p:freeData()
        pd.Ap_X:freeData()
        pd.CtC:freeData()
        pd.SSq:freeData()
        pd.preconditioner:freeData()
        pd.prevX:freeData()

        [maybe_emit(problemSpec:RequiresJtJMaterialize(), quote pd.JTJ:freeData() end)]

        [util.freePrecomputedImages(`pd.parameters,problemSpec)]

        cd(C.cudaFree([&opaque](pd.scanAlphaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanBetaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanAlphaDenominator)))
        cd(C.cudaFree([&opaque](pd.modelCost)))

        cd(C.cudaFree([&opaque](pd.scratch)))
        cd(C.cudaFree([&opaque](pd.q)))
        var hd = pd.hd 
        escape
            if use_cublas or (use_cusparse and materialized_is_dense) then
                emit quote 
                    cb(CUBLAS.cublasDestroy_v2(hd.blas_handle))
                end
            end
            if use_cublas then
                emit quote 
                    cd(C.cudaFree([&opaque](pd.cublas_jtj)))
                end
                if use_direct_solve then
                    emit quote
                        cd(C.cudaFree(hd.cublasSolveMatPtr))
                        cd(C.cudaFree(hd.cublasSolveVecPtr))
                        cd(C.cudaFree(hd.cublasSolveSolutionPtr))
                        cd(C.cudaFree(hd.cublasSolveInvMat))
                        cd(C.cudaFree(hd.cublas_infoArray))
                        cd(C.cudaFree(hd.cublas_PivotArray))
                    end
                end
            end
            if use_cusparse then
                emit quote
                    cs(CUsp.cusparseDestroyMatDescr( hd.cusp_desc ))
                    cs(CUsp.cusparseDestroy( hd.cusp_handle ))
                    cd(C.cudaFree(hd.cusparse_Jp))
                    cd(C.cudaFree(hd.cusparse_permutation))
                    cd(C.cudaFree(pd.cusparse_unsortedJ))
                    if hd.cusparse_JTJ.rowPtr ~= nil then
                        freeCSRMatrix(&hd.cusparse_JTJ)
                    end
                    freeCSRMatrix(&pd.cusparse_J)
                    freeCSRMatrix(&hd.cusparse_JT)
                end
            end
            if problemSpec:RequiresSeparateJtAndJ() then
                emit `pd.Jp:freeData()
            end
        end

        pd.hd:delete()
        -- TODO: Rearchitect to enable deleting of the plan data
        logTrace("Plan freed\n")
    end

    local terra makePlan() : &thallo.Plan
        logTrace("makePlan()\n")
        var pd = PlanData.alloc()
        pd.hd = HostData.alloc()
        var plan : &thallo.Plan = &pd.hd.plan
        plan.data = pd

        plan.init,plan.step,plan.cost,plan.free = init,step,cost,free
        plan.get_summary,plan.reset_unknowns,plan.setsolverparameter,plan.getsolverparameter = get_summary,reset_unknowns,setSolverParameter,getSolverParameter
        plan.estimated_cost = estimated_cost
        logTrace("init GPU vectors\n")
        pd.delta:initGPU()
        pd.r:initGPU()
        pd.b:initGPU()
        pd.Adelta:initGPU()
        pd.z:initGPU()
        pd.p:initGPU()
        pd.Ap_X:initGPU()
        pd.CtC:initGPU()
        pd.SSq:initGPU()
        pd.preconditioner:initGPU()
        pd.initX:initGPU()
        pd.prevX:initGPU()

        [maybe_emit(problemSpec:RequiresJtJMaterialize(), quote pd.JTJ:initGPU() end)]

        logTrace("initializeSolverParameters()\n")
        initializeSolverParameters(&pd.hd.solverparameters)

        [util.initPrecomputedImages(`pd.parameters,problemSpec)]
        logTrace("init GPU scalars\n")	
        cd(C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(thallo_float)))
        cd(C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(thallo_float)))
        cd(C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(thallo_float)))
        cd(C.cudaMalloc([&&opaque](&(pd.modelCost)), sizeof(thallo_float)))

        cd(C.cudaMalloc([&&opaque](&(pd.scratch)), sizeof(thallo_float)))
        cd(C.cudaMalloc([&&opaque](&(pd.q)), sizeof(thallo_float)))

        var hd = pd.hd
        escape
            if use_cublas or (use_cusparse and materialized_is_dense) then
                emit quote
                    cb(CUBLAS.cublasCreate_v2(&(hd.blas_handle)))
                    cb(CUBLAS.cublasSetAtomicsMode(hd.blas_handle, CUBLAS.CUBLAS_ATOMICS_ALLOWED))
                    C.printf("Initialized CUBLAS!\n ")
                end
            end
            if use_cublas then
                emit quote
                    cd(C.cudaMalloc([&&opaque](&(pd.cublas_jtj)), sizeof(thallo_float)*[nUnknowns*nUnknowns]))
                    cd(C.cudaMemsetAsync([&opaque](pd.cublas_jtj), 0, sizeof(thallo_float)*[nUnknowns*nUnknowns], nil))
                end
                if use_direct_solve then
                    emit quote
                        logTrace("Before extra mallocs\n")
                        cd(C.cudaMalloc([&&opaque](&(hd.cublasSolveMatPtr)), sizeof([&thallo_float])))
                        cd(C.cudaMalloc([&&opaque](&(hd.cublasSolveVecPtr)), sizeof([&thallo_float])))
                        cd(C.cudaMalloc([&&opaque](&(hd.cublasSolveSolutionPtr)), sizeof([&thallo_float])))

                        cd(C.cudaMalloc([&&opaque](&(hd.cublasSolveInvMat)), sizeof(thallo_float)*[nUnknowns*nUnknowns]))
                        cd(C.cudaMalloc([&&opaque](&(hd.cublas_infoArray)), sizeof(int)))
                        cd(C.cudaMalloc([&&opaque](&(hd.cublas_PivotArray)), sizeof(int)*[nUnknowns]))

                        logTrace("Before extra memcpies\n")
                        cd(C.cudaMemcpy(hd.cublasSolveMatPtr, &pd.cublas_jtj, sizeof([&thallo_float]), C.cudaMemcpyHostToDevice))
                        cd(C.cudaMemcpy(hd.cublasSolveVecPtr, &pd.p._contiguousallocation, sizeof([&thallo_float]), C.cudaMemcpyHostToDevice))
                        cd(C.cudaMemcpy(hd.cublasSolveSolutionPtr, &hd.cublasSolveInvMat, sizeof([&thallo_float]), C.cudaMemcpyHostToDevice))
                    end
                end
            end
            if use_cusparse then
                emit quote
                    cs(CUsp.cusparseCreate( &hd.cusp_handle ))
                    cs(CUsp.cusparseCreateMatDescr( &hd.cusp_desc ))
                    cs(CUsp.cusparseSetMatType( hd.cusp_desc,CUsp.CUSPARSE_MATRIX_TYPE_GENERAL ))
                    cs(CUsp.cusparseSetMatIndexBase( hd.cusp_desc,CUsp.CUSPARSE_INDEX_BASE_ZERO ))
                    var nnz = int(nnzMaterializedExp)
                    var nResiduals = int(nResidualsMaterializedExp)
                    logSolver("nnz = %s\n",[tostring(nnzMaterializedExp)])
                    logSolver("nResiduals = %s\n",[tostring(nResidualsMaterializedExp)])
                    logSolver("nnz = %d, nResiduals = %d\n",nnz,nResiduals)
                    
                    -- J alloc
                    initCSRMatrix(&pd.cusparse_J,nnz,nResiduals)
                    initCSRMatrix(&hd.cusparse_JT,nnz,int(nUnknowns))
                    
                    
                    -- Jp alloc
                    cd(C.cudaMalloc([&&opaque](&(hd.cusparse_Jp)), sizeof(thallo_float)*nResiduals))
                    cd(C.cudaMalloc([&&opaque](&(hd.cusparse_permutation)), sizeof(int)*nnz))
                    cd(C.cudaMalloc([&&opaque](&(pd.cusparse_unsortedJ)), sizeof(thallo_float)*nnz))

                    -- write J.rowPtr end
                    hd.cusparse_JTJ.rowPtr = nil -- Signal JTJ needs to be allocated

                    C.printf("setting J.rowPtr[%d] = %d\n",nResiduals,nnz)
                    cd(C.cudaMemcpy(&pd.cusparse_J.rowPtr[nResiduals],&nnz,sizeof(int),C.cudaMemcpyHostToDevice))
                    --logDebugCudaThalloIntBuffer("init J.rowPtr", pd.cusparse_J.rowPtr, nResiduals+1)
                end
            end
            if problemSpec:RequiresSeparateJtAndJ() then
                emit `pd.Jp:initGPU()
            end
        end

        logTrace("makePlan() end\n")
        return plan
    end
    return makePlan
end

--util = require("util")

local function is_array(t)
  local i = 0
  for _ in pairs(t) do
      i = i + 1
      if t[i] == nil then return false end
  end
  return true
end
_L = nil
return function(P)
    local L = {}
    local energy = {}
    local residual_group_count = 1
    L.inf = math.huge

    function L.Residuals(tbl)
        local residuals = {}
        for k,v in pairs(tbl) do
            residual_group_count = residual_group_count + 1
            assert(not residuals[k])
            residuals[k] = terralib.newlist()
            if is_array(v) and not ad.ExpVector:isclassof(v) then
                for _,r in ipairs(v) do
                    residuals[k]:insert(r)
                end
            else
                -- TODO: check if expression
                residuals[k]:insert(v)
            end
        end
        energy = P:Energy(residuals)
        return energy
    end

    function L.Schedule(name, jtjp_schedule, compute_at_output, sparse_matrices, compute_lanes)
        assert(energy[name], "Created schedule for nonexistent residual "..name)
        energy[name]:set_jtjp_schedule(jtjp_schedule)
        print("Using old style scheduling, deprecated very soon")
    end

    function L.Dims(...)
        local dims = {}
        for i,name in ipairs {...} do
            dims[i] = thallo.Dim(name,i-1)
        end
        return unpack(dims)
    end

    function L.Result(exauto_index,lin_iter_hint) 
        return P:Cost(energy,exauto_index,lin_iter_hint)
    end

    function L.All(v)
        local r = 1
        for i = 0,v:size() - 1 do
            r = r * v(i)
        end
        return r
    end

    function L.Reduce(fn,init)
        return function(...)
            local r = init
            for _,e in ipairs {...} do
                r = fn(r,e)
            end
            return r
        end
    end
    L.And = L.Reduce(1,ad.and_)
    L.Or = L.Reduce(0,ad.or_)
    L.Not = ad.not_
    
    function L.UsePreconditioner(...) return P:UsePreconditioner(...) end

    function L.gemv(matrix,v)
        local result = terralib.newlist()
        local col_count = v:size()
        local row_count = matrix:size() / col_count
        for r=0,row_count-1 do
            local val = matrix(r*col_count)*v(0)
            for c=1,col_count-1 do
                val = val + matrix(r*col_count+c)*v(c)
            end
            result:insert(val)
        end
        return ad.Vector(unpack(result))
    end

    function L.dot(v0,v1)
        return v0:dot(v1)
    end

    function L.Sqrt(v)
        return ad.sqrt(v)
    end

    function L.normalize(v)
        return v / ad.sqrt(L.dot(v, v))
    end

    function L.length(v0, v1) 
        local diff = v0 - v1
        return ad.sqrt(L.dot(diff, diff))
    end

    function L.Slice(im,s,e)
        return setmetatable({},{
            __call = function(self,ind)
                if s + 1 == e then return im(ind)(s) end
                local t = terralib.newlist()
                for i = s,e - 1 do
                    local val = im(ind)
                    local chan = val(i)
                    t:insert(chan)
                end
                return ad.Vector(unpack(t))
            end })
    end

    function L.Rotate3D(a,v)
        local alpha, beta, gamma = a(0), a(1), a(2)
        local  CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma = ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma)
        local matrix = ad.Vector(
            CosGamma*CosBeta, 
            -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
            SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
            SinGamma*CosBeta,
            CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
            -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
            -SinBeta,
            CosBeta*SinAlpha,
            CosBeta*CosAlpha)
        return L.gemv(matrix,v)
    end
    function L.Rotate2D(angle, v)
	    local CosAlpha, SinAlpha = ad.cos(angle), ad.sin(angle)
        local matrix = ad.Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	    return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
    end
    L.Index = ad.Index
    L.SampledImage = ad.sampledimage
    L.SampledImageArray = ad.sampledimagearray
    function L.Sum(...) return P:TensorContraction(...) end
--
    function L.L_2_norm(v)
        -- TODO: check if scalar and just return
        if ad.ExpVector:isclassof(v) and v:size() > 1 then
            return ad.sqrt(v:dot(v))
        else
            return v
        end
    end
    L.L_p_counter = 1
    function L.L_p(val, p, domains)
        local dist = L.L_2_norm(val)
        local eps = 0.0000001
        local C = ad.pow(dist+eps,(p-2))
        local sqrtC = ad.sqrt(C)
        print("Renable ComputedArray for L_p")
        --[[
        local sqrtCImage = L.ComputedArray("L_p"..tostring(L.L_p_counter),domains,L.Constant(sqrtC))
        L.L_p_counter = L.L_p_counter + 1
        return sqrtCImage(unpack(domains))*val
        --]]
        return L.Constant(sqrtC):get(unpack(domains))*val
    end

    function L.L_1_norm(v)
        -- TODO: check if scalar and just return
        if ad.ExpVector:isclassof(v) and v:size() > 1 then
            local result = 0.0
            for i=1,v:size() do
                result = result + ad.abs(v(i-1))
            end
            return result
        else
            return ad.abs(v)
        end
    end

    function L.L_1(val,domains)
        local dist = L.L_1_norm(val)
        local eps = 0.0000001
        local C = ad.pow(dist+eps,-1)
        local sqrtC = ad.sqrt(C)
        return L.Constant(sqrtC)*dist
    end

    L.Select = ad.select

    L.Constant = ad.constant

    function L.SelectOnAll(pList,val,default)
        assert(#pList > 0, "SelectOnAll() requires at least one predicate")
        local result = L.Select(pList[#pList], val, default)
        for i=1,#pList-1 do
            local p = pList[#pList-i]
            result = L.Select(p, result, default)
        end
        return result
    end

    --TODO: Check transpose is correct
    function L.RodriguesSO3Exp(w, A, B)
        local R00, R01, R02
        local R10, R11, R12
        local R20, R21, R22
        do
            local wx2 = w(0) * w(0);
            local wy2 = w(1) * w(1);
            local wz2 = w(2) * w(2);
            R00 = 1.0 - B*(wy2 + wz2);
            R11 = 1.0 - B*(wx2 + wz2);
            R22 = 1.0 - B*(wx2 + wy2);
        end
        do
            local a = A*w(2)
            local b = B*(w(0) * w(1))
            R01 = b - a;
            R10 = b + a;
        end
        do
            local a = A*w(1)
            local b = B*(w(0) * w(2))
            R02 = b + a;
            R20 = b - a;
        end
        do
            local a = A*w(0)
            local b = B*(w(1) * w(2))
            R12 = b - a;
            R21 = b + a;
        end
        return ad.Vector(   R00, R01, R02,
                            R10, R11, R12,
                            R20, R21, R22)
    end

    function L.cross(a,b)
        return ad.Vector(a(1)*b(2) - a(2)*b(1), a(2)*b(0) - a(0)*b(2), a(0)*b(1) - a(1)*b(0))
    end

    function L.Matrix4(...)
        assert(select("#",...) == 16, "Provided "..tostring(select("#",...)).." elements to Matrix4 constructor, need 16")
        return ad.Vector(...)
    end

    function L.Vec4(...)
        assert(select("#",...) == 4, "Provided "..tostring(select("#",...)).." elements to Vec4 constructor, need 4")
        return ad.Vector(...)
    end 

    function L.RotationMatrixAndTranslationToMat4(r, t)
        return ad.Vector(  r(0), r(1), r(2), t(0),
                    r(3), r(4), r(5), t(1),
                    r(6), r(7), r(8), t(2),
                     0.0,  0.0,  0.0, 1.0)
    end

    function L.Mat4ToRigidTransform(m)
        return ad.Vector(  m(0), m(1), m(2), m(3),
                           m(4), m(5), m(6), m(7),
                           m(8), m(9), m(10), m(11))
    end

    function L.RigidTransformToMat4(m)
        return ad.Vector(  m(0), m(1), m(2), m(3),
                           m(4), m(5), m(6), m(7),
                           m(8), m(9), m(10), m(11),
                           0.0, 0.0, 0.0, 1.0)
    end

    function L.CameraToDepth(fx, fy, cx, cy, pos)
        return ad.Vector(
            pos(0)*fx / pos(2) + cx,
            pos(1)*fy / pos(2) + cy)
    end

    function L.Max(a,b)
        return ad.select(ad.greater(a,b),a,b)
    end

    -- TODO: check transpose
    function L.matmul(a,b)
        local result = terralib.newlist()
        local dim = math.sqrt(a:size())
        assert(a:size() == b:size() and dim == math.floor(dim), "gemm currently only implemented for square matrices of the same size, but given sizes "..
                    tostring(a:size()).." and "..tostring(b:size()))
        -- c[i,j] = +[k](a[i,k]*b[k,j])
        for i=0,dim-1 do
            for j=0,dim-1 do
                local c = 0.0
                for k=0,dim-1 do
                    c = c + (a(i*dim+k)*b(k*dim+j))
                end
                result:insert(c)
            end
        end
        return ad.Vector(unpack(result))
    end

    function L.InverseMatrix4(entries)
        local inv = {}
        inv[0] = entries[5]  * entries[10] * entries[15] - 
            entries[5]  * entries[11] * entries[14] - 
            entries[9]  * entries[6]  * entries[15] + 
            entries[9]  * entries[7]  * entries[14] +
            entries[13] * entries[6]  * entries[11] - 
            entries[13] * entries[7]  * entries[10]

        inv[4] = -entries[4]  * entries[10] * entries[15] + 
            entries[4]  * entries[11] * entries[14] + 
            entries[8]  * entries[6]  * entries[15] - 
            entries[8]  * entries[7]  * entries[14] - 
            entries[12] * entries[6]  * entries[11] + 
            entries[12] * entries[7]  * entries[10]

        inv[8] = entries[4]  * entries[9] * entries[15] - 
            entries[4]  * entries[11] * entries[13] - 
            entries[8]  * entries[5] * entries[15] + 
            entries[8]  * entries[7] * entries[13] + 
            entries[12] * entries[5] * entries[11] - 
            entries[12] * entries[7] * entries[9]

        inv[12] = -entries[4]  * entries[9] * entries[14] + 
            entries[4]  * entries[10] * entries[13] +
            entries[8]  * entries[5] * entries[14] - 
            entries[8]  * entries[6] * entries[13] - 
            entries[12] * entries[5] * entries[10] + 
            entries[12] * entries[6] * entries[9]

        inv[1] = -entries[1]  * entries[10] * entries[15] + 
            entries[1]  * entries[11] * entries[14] + 
            entries[9]  * entries[2] * entries[15] - 
            entries[9]  * entries[3] * entries[14] - 
            entries[13] * entries[2] * entries[11] + 
            entries[13] * entries[3] * entries[10]

        inv[5] = entries[0]  * entries[10] * entries[15] - 
            entries[0]  * entries[11] * entries[14] - 
            entries[8]  * entries[2] * entries[15] + 
            entries[8]  * entries[3] * entries[14] + 
            entries[12] * entries[2] * entries[11] - 
            entries[12] * entries[3] * entries[10]

        inv[9] = -entries[0]  * entries[9] * entries[15] + 
            entries[0]  * entries[11] * entries[13] + 
            entries[8]  * entries[1] * entries[15] - 
            entries[8]  * entries[3] * entries[13] - 
            entries[12] * entries[1] * entries[11] + 
            entries[12] * entries[3] * entries[9]

        inv[13] = entries[0]  * entries[9] * entries[14] - 
            entries[0]  * entries[10] * entries[13] - 
            entries[8]  * entries[1] * entries[14] + 
            entries[8]  * entries[2] * entries[13] + 
            entries[12] * entries[1] * entries[10] - 
            entries[12] * entries[2] * entries[9]

        inv[2] = entries[1]  * entries[6] * entries[15] - 
            entries[1]  * entries[7] * entries[14] - 
            entries[5]  * entries[2] * entries[15] + 
            entries[5]  * entries[3] * entries[14] + 
            entries[13] * entries[2] * entries[7] - 
            entries[13] * entries[3] * entries[6]

        inv[6] = -entries[0]  * entries[6] * entries[15] + 
            entries[0]  * entries[7] * entries[14] + 
            entries[4]  * entries[2] * entries[15] - 
            entries[4]  * entries[3] * entries[14] - 
            entries[12] * entries[2] * entries[7] + 
            entries[12] * entries[3] * entries[6]

        inv[10] = entries[0]  * entries[5] * entries[15] - 
            entries[0]  * entries[7] * entries[13] - 
            entries[4]  * entries[1] * entries[15] + 
            entries[4]  * entries[3] * entries[13] + 
            entries[12] * entries[1] * entries[7] - 
            entries[12] * entries[3] * entries[5]

        inv[14] = -entries[0]  * entries[5] * entries[14] + 
            entries[0]  * entries[6] * entries[13] + 
            entries[4]  * entries[1] * entries[14] - 
            entries[4]  * entries[2] * entries[13] - 
            entries[12] * entries[1] * entries[6] + 
            entries[12] * entries[2] * entries[5]

        inv[3] = -entries[1] * entries[6] * entries[11] + 
            entries[1] * entries[7] * entries[10] + 
            entries[5] * entries[2] * entries[11] - 
            entries[5] * entries[3] * entries[10] - 
            entries[9] * entries[2] * entries[7] + 
            entries[9] * entries[3] * entries[6]

        inv[7] = entries[0] * entries[6] * entries[11] - 
            entries[0] * entries[7] * entries[10] - 
            entries[4] * entries[2] * entries[11] + 
            entries[4] * entries[3] * entries[10] + 
            entries[8] * entries[2] * entries[7] - 
            entries[8] * entries[3] * entries[6]

        inv[11] = -entries[0] * entries[5] * entries[11] + 
            entries[0] * entries[7] * entries[9] + 
            entries[4] * entries[1] * entries[11] - 
            entries[4] * entries[3] * entries[9] - 
            entries[8] * entries[1] * entries[7] + 
            entries[8] * entries[3] * entries[5]

        inv[15] = entries[0] * entries[5] * entries[10] - 
            entries[0] * entries[6] * entries[9] - 
            entries[4] * entries[1] * entries[10] + 
            entries[4] * entries[2] * entries[9] + 
            entries[8] * entries[1] * entries[6] - 
            entries[8] * entries[2] * entries[5]

        local matrixDet = entries[0] * inv[0] + entries[1] * inv[4] + entries[2] * inv[8] + entries[3] * inv[12];

        local d_r = 1.0 / matrixDet;
        local res = {}
        for i=1,16 do
            res[i] = inv[i-1]*d_r
        end
        return L.Matrix4(unpack(res))
    end

    function L.rotationFromMat4(t)
        return ad.Vector(
            t(0), t(1), t(2),
            t(4), t(5), t(6),
            t(8), t(9), t(10)
            )
    end

    function L.translationFromMat4(t)
        return ad.Vector(t(3), t(7), t(11))
    end

    function L.transpose(M)
        local result = terralib.newlist()
        local dim = math.sqrt(M:size())
        assert(dim == math.floor(dim), "transpose currently only implemented for square matrices, but given size "..
                    tostring(M:size()))
        for i=0,dim-1 do
            for j=0,dim-1 do
                result:insert(M(j*dim+i))
            end
        end
        return ad.Vector(unpack(result))
    end

    function L.InvertRigidTransform(transform)
        local R = L.rotationFromMat4(transform)
        local t = L.translationFromMat4(transform)
        local Rt = L.transpose(R)
        local newT = L.gemv(-Rt,t)
        return L.Matrix4(
            Rt(0),Rt(1),Rt(2),newT(0),
            Rt(3),Rt(4),Rt(5),newT(1),
            Rt(6),Rt(7),Rt(8),newT(2),
            0,0,0,1)
    end

    function L.PoseToMatrix(rot,trans)
        local theta_sq = L.dot(rot, rot);
        local theta = L.Sqrt(theta_sq);

        local cr = L.cross(rot, trans);
        local smallAngle = ad.less(theta_sq, 1e-8)

        local ONE_SIXTH = (1.0/6.0)
        local ONE_TWENTIETH = (1.0/20.0)
        
        local A_s = 1.0 - ONE_SIXTH * theta_sq
        local B_s = 0.5
        local translation_s = trans + 0.5 * cr;

        local midAngle = ad.less(theta_sq, 1e-6)
        local C_m = ONE_SIXTH*(1.0 - ONE_TWENTIETH * theta_sq)
        local A_m = 1.0 - theta_sq * C_m
        local B_m = 0.5 - (0.25 * ONE_SIXTH * theta_sq)
        local inv_theta = 1.0 / theta
        local A_l = ad.sin(theta) * inv_theta
        local B_l = (1.0 - ad.cos(theta)) * (inv_theta * inv_theta)
        local C_l = (1.0 - A_l) * (inv_theta * inv_theta)
        local w_cross = L.cross(rot, cr)

        local translation_m = trans + B_m * cr + C_m * w_cross
        local translation_l = trans + B_l * cr + C_l * w_cross

        local translation = ad.select(smallAngle, translation_s, ad.select(midAngle, translation_m, translation_l))
        local A = ad.select(smallAngle, A_s, ad.select(midAngle, A_m, A_l))
        local B = ad.select(smallAngle, B_s, ad.select(midAngle, B_m, B_l))

        -- 3x3 rotation part:
        local rotationMatrix = L.RodriguesSO3Exp(rot, A, B)

        return L.RotationMatrixAndTranslationToMat4(rotationMatrix, translation)
    end


    function L.Vec3(v)
        return ad.Vector(v(0),v(1),v(2))
    end

    function L.rigid_trans(M,v)
        return L.Vec3(L.gemv(M,ad.Vector(v(0),v(1),v(2),1.0)))
    end


    -- Adapted from Ceres (rotation.h)
    function L.AngleAxisRotatePoint(angle_axis, pt)
        local theta2 = L.dot(angle_axis,angle_axis)
        -- std::numeric_limits<double>::epsilon() in Ceres, we use our own cutoff here
        local large_axis = ad.greater(theta2, 1e-8)

        -- Away from zero, use the rodriguez formula
        --
        --   result = pt costheta +
        --            (w x pt) * sintheta +
        --            w (w . pt) (1 - costheta)
        --
        -- We want to be careful to only evaluate the square root if the
        -- norm of the angle_axis vector is greater than zero. Otherwise
        -- we get a division by zero.
        local theta = ad.sqrt(theta2);
        local costheta = ad.cos(theta);
        local sintheta = ad.sin(theta);
        local theta_inverse = 1.0 / theta;
        local w = angle_axis*theta_inverse
        local w_cross_pt = L.cross(w,pt)
        local tmp = L.dot(w,pt) * (1.0 - costheta)
        local large_result = pt*costheta + w_cross_pt*sintheta + w*tmp

        -- Near zero, the first order Taylor approximation of the rotation
        -- matrix R corresponding to a vector w and angle w is
        --
        --   R = I + hat(w) * sin(theta)
        --
        -- But sintheta ~ theta and theta * w = angle_axis, which gives us
        --
        --  R = I + hat(w)
        --
        -- and actually performing multiplication with the point pt, gives us
        -- R * pt = pt + w x pt.
        --
        -- Switching to the Taylor expansion near zero provides meaningful derivatives

        local w_cross_pt_s = L.cross(angle_axis,pt)
        local small_result = pt + w_cross_pt_s
        -- Choose based on axis length
        return ad.select(large_axis, large_result, small_result)
    end



    function L.Stencil (lst)
        local i = 0
        return function()
            i = i + 1
            if not lst[i] then return nil
            else return unpack(lst[i]) end
        end
    end

    function L.Sparse(...) return {name="Sparse",args={...}} end
    function L.Array(...) return {name="Array",args={...}} end
    function L.Unknown(...) return {name="Unknown",args={...}} end
    function L.Param(...) return {name="Param",args={...}} end

    function L.Image(...) 
        print("Warning: 'Image' syntax deprecated; use 'Array' instead")
        return {name="Array",args={...}} 
    end

    function L.Inputs(tbl)
        for k,v in pairs(tbl) do
            L[k] = P[v.name](P,k,unpack(v.args))
        end
    end

    setmetatable(L,{__index = function(self,key)
        if type(P[key]) == "function" then
            return function(...) return P[key](P,...) end
        end
        if key ~= "select" and ad[key] then return ad[key] end
        if thallo[key] then return thallo[key] end
        return _G[key]
    end})
    _G._L = L
    return L
end
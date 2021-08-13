local ad = {}
local C = terralib.includec("math.h")
local A = require("asdl").NewContext()
require("precision")
local use_simplify = true
local use_condition_factoring = true
local use_polysimplify = true
local List = terralib.newlist

local function rPrint(s, l, i) -- recursive Print (structure, limit, indent)
    l = (l) or 100; i = i or "";    -- default item limit, indent string
    local ts = type(s);
    if (l<1) then print (i,ts," *snip* "); return end;
    if (ts ~= "table") then print (i,ts,s); return end
    print (i,ts);           -- print "table"
    for k,v in pairs(s) do  -- print "[KEY] VALUE"
        rPrint(v, l-1, i.."\t["..tostring(k).."]");
    end
end 


A:Extern("TerraType",terralib.types.istype)
A:Define [[
    Op = (string name)
    ExpVector = (table* data) unique
    Exp = Var(any key_) unique
        | Apply(Op op, number? const, Exp* args) unique
        | Const(number v) unique
]]
local Op,ExpVector,Exp,Var,Apply,Const = A.Op,A.ExpVector,A.Exp,A.Var,A.Apply,A.Const
ad.classes = A

local nextid = 0
local function allocid()
    nextid = nextid + 1
    return nextid
end

local function sortexpressions(es)
    local function order(a,b)
        return a.id < b.id
    end
    table.sort(es,order)
    return es
end

local empty = terralib.newlist {}
function Exp:children() return empty end
function Apply:children() return self.args end

function Exp:type()
    assert(self.type_ == bool or self.type_ == thallo_float) 
    return self.type_ 
end

function Const:__tostring() return tostring(self.v) end

function Var:init()
    local key = self.key_
    self.id = allocid()
    self.type_ = thallo_float
    if type(key) == "table" and type(key.type) == "function" then
        self.type_ = key:type()
    end 
    assert(self.type_ == thallo_float or self.type_ == bool, "variable with key exists with a different type")
end

function Apply:init()
    assert(not self.op.nparams or #self.args == self.op.nparams)
    self.type_ = self.op:propagatetype(self.args)
    self.id = allocid()
end

function Const:init()
    self.id = allocid()
    self.type_ = thallo_float
end

local function toexp(n)
    if n then 
        if Exp:isclassof(n) then 
            local mt = getmetatable(n)
            return n
        elseif ExpVector:isclassof(n) and #n.data == 1 then return n.data[1]
        elseif type(n) == "number" then return Const(n)
        end
    end
    return nil
end
local zero,one,negone = toexp(0),toexp(1),toexp(-1)
local function allconst(args)
    for i,a in ipairs(args) do
        if not Const:isclassof(a) then return false end
    end
    return true
end

function Var:key() return self.key_ end

--[[
  cases: sum + sum -> turn each child into a factor (c,[t]) and join
  cases: sum + prod -> use sum + sum
  cases: sum + pow -> use sum + sum
  
  cases: prod*prod -> new factor with combined terms turn each term into pow (c,t) (done)
  cases: prod*pow -> use prod*prod (done)
  
  cases: prod*sum -> distribute prod into sum
]]

local function asprod(exp)
    if Apply:isclassof(exp) and exp.op.name == "prod" then
        return exp.const, exp:children()
    elseif Const:isclassof(exp) then
        return exp.v, empty
    else
        return 1.0,terralib.newlist { exp }
    end
end
local function aspowc(exp)
    if Apply:isclassof(exp) and exp.op.name == "powc" then
        return exp.const, exp:children()[1]
    else
        return 1.0,exp
    end
end

local function orderedexpressionkeys(tbl)
    local keys = terralib.newlist()
    for k,v in pairs(tbl) do
        keys:insert(k)
    end
    sortexpressions(keys)
    return keys
end
    
local function simplify(self)
    local op,const,args = self.op,self.const,self.args
    if not use_simplify then return self end
    
    if allconst(args) then
        local r = op:propagateconstant(args:map("v"))
        if r then return r end
    end
    
    if op.name == "sum" then
        local root = {}
        local function lookup(es)
            local node = root
            for _,e in ipairs(es) do
                local next = node[e]
                if not next then
                    next = {}
                    node[e] = next
                end
                node = next
            end
            return assert(node)
        end
        
        local termnodes = terralib.newlist()
        
        local function insertall(args)
            for i,a in ipairs(args) do
                if Const:isclassof(a) then
                    const = const + a.v
                elseif Apply:isclassof(a) and a.op.name == "sum" then
                    const = const + a.const
                    insertall(a.args)
                else
                    local c,aa = asprod(a)
                    local tbl = lookup(aa)
                    if not tbl.value then
                        tbl.c,tbl.value = 0,aa
                        termnodes:insert(tbl)
                    end
                    tbl.c = tbl.c + c
                end
            end
        end
        insertall(args)
    
        if #termnodes == 0 then return toexp(const) end
        local terms = terralib.newlist()
        for _,t in ipairs(termnodes) do
            if t.c ~= 0 then
                terms:insert(ad.prod(t.c,unpack(t.value)))
            end
        end
        if const == 0.0 and #terms == 1 then return terms[1] end
        return Apply(op,const,sortexpressions(terms))
    elseif op.name == "prod" then
        local expmap = {} -- maps each term to the power it has
        local function insertall(args)
            for i,a in ipairs(args) do
                local c, es = asprod(a)
                const = const * c
                for _,e in ipairs(es) do
                    local c,ep = aspowc(e)
                    expmap[ep] = (expmap[ep] or 0) + c
                end
            end
        end
        insertall(args)
        if const == 0.0 then return zero end
        
        local factors = terralib.newlist()
        local keys = orderedexpressionkeys(expmap)
        for _,k in ipairs(keys) do
            local v = expmap[k]
            if v ~= 0 then
                factors:insert(k^v)
            end
        end
        
        if #factors == 0 then return toexp(const) end
        if #factors == 1 and const == 1.0 then return factors[1] end
        return Apply(op,const,sortexpressions(factors))
    end
    
    local x,y,z = unpack(args)
    
    if op.name == "pow" then
        -- TODO: specialize for non-integral powers
        if Const:isclassof(y) then
            if (y.v == math.floor(y.v)) then
                if y.v == 1.0 then
                    return x
                elseif y.v == 0.0 then
                    return one
                else
                    local c,xx = aspowc(x)
                    return ad.powc(y.v*c,xx)
                end
            else
                print("non-integral pow "..tostring(y.v)..". Not specializing.")
            end
        end
    elseif op.name == "powc" then
        if x:type() == bool then
            return x
        elseif Apply:isclassof(x) and x.op.name == "sqrt" and const == 2 then
            return x.args[1]
        end
    elseif op.name == "select" then
        if Const:isclassof(x) then
            return  x.v ~= 0 and y or z
        elseif y == zero then
            return ad.not_(x) * z
        elseif z == zero then
            return x * y
        end
    elseif op.name == "or_" then
        if x == y then return x
        elseif Const:isclassof(x) then
            if x.v ~= 0 then return one
            else return y end
        elseif Const:isclassof(y) then
            if y.v  ~= 0 then return one
            else return x end
        end
    end
    return self
end

function Apply:simplified()
    if not self._simplified then
        self._simplified = simplify(self)
    end
    return self._simplified 
end

function ExpVector:size() return #self.data end
function ExpVector:__tostring() return "{"..ad.tostrings(self.data).."}" end
function ExpVector:__index(key)
    if type(key) == "number" then
        assert(key >= 0 and key < #self.data, "index out of bounds")
        return self.data[key+1]
    else return ExpVector[key] end
end
ExpVector.__call = ExpVector.__index
local function toexpvectorentry(v)
    return ExpVector:isclassof(v) and v or toexp(v)
end
function ExpVector:__newindex(key,v)
    assert(type(key) == "number", "unknown field in ExpVector: "..tostring(key))
    assert(key >= 0 and key < #self.data, "index out of bounds")
    self.data[key+1] = assert(toexpvectorentry(v), "expected a ExpVector or a valid expression")
end
function ExpVector:map(fn)
    return ad.Vector(unpack(self.data:map(fn)))
end
function ExpVector:expressions() return self.data end    
function ExpVector:sum()
    local s = 0
    for i,e in ipairs(self:expressions()) do
        s = s + e
    end
    return s
end
function ExpVector:dot(rhs)
    return (self*rhs):sum()
end

function ExpVector:slice(start,finish)
    local data = terralib.newlist()
    for i = start+1,finish do
        data[i-start] = assert(toexpvectorentry(self.data[i]),"expected a ExpVector or valid expression")
    end
    return ExpVector(data)
end

-- generates variable names
local v = {} 
setmetatable(v,{__index = function(self,key)
    local r = Var(key)
    v[key] = r
    return r
end})

local x,y,z = v[1],v[2],v[3]

ad.v = v
ad.toexp = toexp

function ad.Vector(...)
    local data = terralib.newlist()
    for i = 1,select("#",...) do
        local e = select(i,...)
        data[i] = assert(toexpvectorentry(e),"expected a ExpVector or valid expression")
    end
    return ExpVector(data)
end
ad.ExpVector = ExpVector

setmetatable(ad, { __index = function(self,idx) -- for just this file, auto-generate an op 
    local name = assert(type(idx) == "string" and idx)
    local op = Op(name)
    rawset(self,idx,op)
    return op
end })

local function conformvectors(args)
    local N
    for i,e in ipairs(args) do
        if ExpVector:isclassof(e) then
            if e:size() == 1 then
                --Ignore "scalars" TODO: redo work to always get rid of length 1 vectors...
                --args[i] = toexp(e[1])
            else
                assert(not N or N == e:size(), "non-conforming vector sizes")
                N = e:size()
            end
        else
            local arg = toexp(e) 
            if not arg then
                assert(arg,"expected an expression, instead found "..tostring(e))
            end
            args[i] = arg
        end
    end
    return N
end

function Op:create(const,args)
    local N = conformvectors(args)
    if not N then return Apply(self,const,args):simplified() end
    local exps = List()
    for i = 0,N-1 do
        local newargs = args:map(function(e) return ExpVector:isclassof(e) and e[i] or e end)
        exps:insert(self:create(const,newargs))
    end
    return ExpVector(exps)
end

function Op:__call(c,...)
    if self.hasconst then return self:create(c,List{...})
    else return self:create(nil,List{c,...}) end
end

local function insertcast(from,to,exp)
    assert(terralib.types.istype(from) and terralib.types.istype(to))
    if from == to then return exp
    else return `to(exp) end
end
local function insertcasts(exp,args)
    local nargs = terralib.newlist()
    local t,ta = exp.op:propagatetype(exp:children()) 
    for i,from in ipairs(exp:children()) do
        nargs[i] = insertcast(from:type(),ta[i],args[i])
    end
    return nargs
end

Op.hasconst = false -- operators that have a constant override this
function Op:define(fn,...)
    local dbg = debug.getinfo(fn,"u")
    assert(not dbg.isvararg)
    self.nparams = dbg.nparams
    function self:generate(exp,args) 
        return fn(unpack(insertcasts(exp,args)))
    end
    self.derivs = terralib.newlist()
    for i = 1,select("#",...) do
        local e = select(i,...)
        self.derivs:insert((assert(toexp(e),"expected an expression")))
    end
    
    local syms,vars = terralib.newlist(),terralib.newlist()
    for i = 1,self.nparams do
        syms:insert(symbol(thallo_float))
        vars:insert(ad.v[i])
    end
    local cpropexpression = self(unpack(vars))
    local r = self:generate(cpropexpression,syms)
    terra self.impl([syms]) return thallo_float(r) end    
    
    return self
end

function Op:propagateconstant(args)
   if not self.impl then return nil end
    return assert(toexp(self.impl(unpack(args))), "result is not an expression")
end

local function rep(N,v) 
    local r = terralib.newlist()
    for i = 1,N do
        r:insert(v)
    end
    return r
end
    
function Op:propagatetype(args) --returns a 2: <returntype>, <castedargumenttypes>
    -- default is 'float', special ops will override this
    return thallo_float, rep(#args,thallo_float)
end


function Op:__tostring() return self.name end

local mo = {"add","sub","mul","div", "pow"}
for i,o in ipairs(mo) do
    local function impl(a,b) return (ad[o])(a,b) end
    Exp["__"..o], ExpVector["__"..o] = impl,impl
end
Exp.__unm = function(a) return ad.unm(a) end
ExpVector.__unm = function(a) return ad.unm(a) end

function Exp:rename(vars)
    local varsf = type(vars) == "function" and vars or function(k) return vars[k] end
    local visitcached
    local function visit(self) 
        if self.kind == "Apply" then
            local nargs = self.args:map(visitcached)
            return self.op:create(self.const,nargs)
        elseif self.kind == "Const" then
            return self
        elseif self.kind == "Var" then
            local nv = toexp(varsf(self:key()))
            if not nv then
                assert(nv,
                          ("rename: unknown invalid mapping for variable %s which maps to %s"):format(tostring(self:key()),tostring(nv)))
            end
            return nv
        end
    end
    local cached = {} 
    function visitcached(self)
        local r = cached[self]
        if r then return r end
        r = visit(self)
        cached[self] = r
        return r
    end
    return visitcached(self)
end

function Exp:visit(fn,data) -- cheaper way to find all the variable keys when you are not modifying them
    local visited = {}
    local function visit(self,data)
        if visited[self] then return end
        visited[self] = true
        if self.kind == "Var" then
            fn(self:key(),data)
        end
        for i,c in ipairs(self:children()) do
            visit(c,data)
        end
    end
    visit(self,data)
end

local function countuses(es)
    local uses = {}
    local function count(e)
        uses[e] = (uses[e] or 0) + 1
        if uses[e] == 1  then
            for i,a in e:children() do count(a) end
        end
    end
    for i,a in ipairs(es) do 
        if ExpVector:isclassof(a) then
            for i,e in ipairs(a:expressions()) do
                count(e)
            end
        else
            count(a)
        end 
    end
    for k,v in pairs(uses) do
       uses[k] = v > 1 or nil
    end
    return uses
end    

local ispoly = {sum = 0, prod = 1, powc = 2}
local function expstostring(es)
    es = (terralib.islist(es) and es) or terralib.newlist(es)
    local linearized = terralib.newlist()
    local numfree = terralib.newlist()
    local uses = {}
    local exptoidx = {}
    local function visit(e)
        if not exptoidx[e] then
            for i,c in ipairs(e:children()) do visit(c) end
            linearized:insert(e)
            exptoidx[e] = #linearized
            uses[e] = 0
        end
        uses[e] = uses[e] + 1
    end
    for i,e in ipairs(es) do 
        visit(e) 
    end
    
-------------------------------
    local exptoreg = {}
    local nextregister = 0
    local freeregisters = terralib.newlist()
    local function releaseregister(i)
        freeregisters:insert(i)
    end
    local function registerforexp(e)
        if e.kind == "Apply" or e.kind == "Const" then return -1 end -- no registers for const/var
        if exptoreg[e] then return exptoreg[e] end
        local r
        if #freeregisters > 0 then 
            r = freeregisters:remove()
        else 
            r = nextregister
            nextregister = nextregister + 1
        end
        exptoreg[e] = r
        return r
    end
---------------------------------
    
    local shouldprint = {}
    local function stringforuse(e)
        shouldprint[e] = true
        local r
        if "Var" == e.kind then
            local k = e:key()
            r = type(k) == "number" and ("v%d"):format(k) or tostring(e:key()) 
        elseif "Const" == e.kind then
            r = tostring(e.v)
        else
            r = ("r%d"):format(exptoidx[e])
        end
        if e:type() == bool then
            r = ("<%s>"):format(r)
        end
        return r
    end    

    local function emitpoly(e,l) 
        if not Apply:isclassof(e) or not ispoly[e.op.name] or l > ispoly[e.op.name] then
            return stringforuse(e)
        end
        if e.op.name == "powc" then
            return ("%s^%s"):format(stringforuse(e.args[1]),e.const)
        elseif e.op.name == "prod" then
            local r = e.args:map(emitpoly,2):concat("*")
            if e.const ~= 1 then
                r = ("%s*%s"):format(e.const,r)
            end
            return r
        elseif e.op.name == "sum" then
            local r = e.args:map(emitpoly,1):concat(" + ")
            if e.const ~= 0 then
                r = ("%s + %s"):format(r,e.const)
            end
            return r
        end
    end
    
    local function emitapp(e)
        if ispoly[e.op.name] then
            return emitpoly(e,0)
        end
        local name = e.op.name
        if e.op.hasconst then
            name = ("%s[%f]"):format(name,e.const)
        end
        return ("%s(%s)"):format(name,e.args:map(stringforuse):concat(","))
    end
    
    local estring = es:map(stringforuse):concat(",")
    
    local tbl = terralib.newlist()
    for i = #linearized,1,-1 do
        local e = linearized[i]
        if e.kind == "Apply" then
            releaseregister(registerforexp(e))
            for i,c in ipairs(e:children()) do
                registerforexp(c)
            end
            if shouldprint[e] then
                local rhs = emitapp(e)
                tbl:insert(("[%2d,%d]  r%d : %s = %s\n"):format(registerforexp(e),e.id,i,e:type(),rhs))
            end
        end
    end
--------------------------------
    local rtbl = terralib.newlist()
    for i = #tbl,1,-1  do
        rtbl:insert(tbl[i])
    end
    if #rtbl == 0 then return estring end
    return ("let (%d reg) \n%sin\n  %s\nend\n"):format(nextregister, rtbl:concat(),estring)
end

ad.tostrings = expstostring
Exp.__tostring = nil -- force overrides to happen
function Exp:__tostring()
    return expstostring(terralib.newlist{self})
end

function Exp:d(v)
    assert(Var:isclassof(v))
    self.derivs = self.derivs or {}
    local r = self.derivs[v]
    if r then return r end
    r = self:calcd(v)
    self.derivs[v] = r
    return r
end

function Exp:partials()
    return empty
end

function Op:getpartials(exp)
    assert(#self.derivs == #exp.args, "number of arguments do not match number of partials")
    return self.derivs:map("rename",exp.args)
end

function Apply:partials()
    self.partiallist = self.partiallist or self.op:getpartials(self)
    return self.partiallist
end


function Var:calcd(v)
    if self == v then
        return one
    end
    local k = self:key()
    if type(k) == "table" and type(k.gradient) == "function" then -- allow variables to express external relationships to other variables
        local gradtable = k:gradient() -- table of (var -> exp) mappings that are the gradient of this variable with respect to all unknowns of interest
        local r = gradtable[v:key()]
        if r then return assert(toexp(r),"expected an ad expression") end
    end
    return zero
end
function Const:calcd(v)
    return zero
end
function Apply:calcd(v)
    local dargsdv = self.args:map("d",v)
    local dfdargs = self:partials()
    local r
    for i = 1,#self.args do
        local e = dargsdv[i]*dfdargs[i]
        r = (not r and e) or (r + e)
    end
    return r
end

--calc d(thisexpress)/d(exps[1]) ... d(thisexpress)/d(exps[#exps]) (i.e. the gradient of this expression with relation to the inputs) 


function Exp:gradient(exps)
    return exps:map(function(v) return self:d(v) end)
end

function ad.sum:generate(exp,args)
    args = insertcasts(exp,args)
    local r = `thallo_float(exp.const)
    for i,c in ipairs(args) do
        r = `r+c
    end
    return r
end
function ad.sum:getpartials(exp) return rep(#exp.args,one) end
ad.sum.hasconst = true

function ad.add(x,y) return ad.sum(0,x,y) end
function ad.sub(x,y) return ad.sum(0,x,-y) end
function ad.mul(x,y) return ad.prod(1,x,y) end
function ad.div(x,y) return ad.prod(1,x,y^-1) end

function ad.prod:generate(exp,args)
    local r = `thallo_float(exp.const)
    local condition = true
    for i,ce in ipairs(exp:children()) do
        local a = args[i]
        if ce:type() == bool then
            condition = `condition and a
        else
            r = `r*a
        end
    end
    if condition == true then
        return r
    else
        return `terralib.select(condition,r,0.f)
    end
end
function ad.prod:getpartials(exp)
    local r = terralib.newlist()
    for i,a in ipairs(exp.args) do
        local terms = terralib.newlist()
        for j,a2 in ipairs(exp.args) do
            if i ~= j then
                terms:insert(a2)
            end
        end
        r:insert(ad.prod(exp.const,unpack(terms)))
    end
    return r
end
ad.prod.hasconst = true

local genpow = terralib.memoize(function(N)
    local terra pow(a : thallo_float) : thallo_float
        var r : thallo_float = [thallo_float](1.f)
        for i = 0,N do
            r = r*a
        end
        return r
    end 
    pow:setname("pow"..tostring(N))
    return pow
end)
function ad.powc:generate(exp,args)
    args = insertcasts(exp,args)
    local c,e = exp.const, args[1]
    if c == 1 then
        return e
    elseif c > 0 then
        return `[genpow(c)](e)
    else
        return `1.f/[genpow(-c)](e)
    end
end

function ad.powc:getpartials(exp)
    local x = exp.args[1]
    local c = exp.const
    return terralib.newlist { c*ad.pow(x,c-1) }
end
ad.powc.hasconst = true

function ad.unm(x) return ad.prod(-1.0,x) end

ad.acos:define(function(x) return `C.acos(x) end, -1.0/ad.sqrt(1.0 - x*x))
ad.acosh:define(function(x) return `C.acosh(x) end, 1.0/ad.sqrt(x*x - 1.0))
ad.asin:define(function(x) return `C.asin(x) end, 1.0/ad.sqrt(1.0 - x*x))
ad.asinh:define(function(x) return `C.asinh(x) end, 1.0/ad.sqrt(x*x + 1.0))
ad.atan:define(function(x) return `C.atan(x) end, 1.0/(x*x + 1.0))
ad.atan2:define(function(x,y) return `C.atan2(x*x+y*y,y) end, y/(x*x+y*y),x/(x*x+y*y))
ad.cos:define(function(x) return `C.cos(x) end, -ad.sin(x))
ad.cosh:define(function(x) return `C.cosh(x) end, ad.sinh(x))
ad.exp:define(function(x) return `C.exp(x) end, ad.exp(x))
ad.log:define(function(x) return `C.log(x) end, 1.0/x)
ad.log10:define(function(x) return `C.log10(x) end, 1.0/(ad.log(10.0)*x))
ad.pow:define(function(x,y) return `C.pow(x,y) end, y*ad.pow(x,y)/x,ad.log(x)*ad.pow(x,y)) 
ad.sin:define(function(x) return `C.sin(x) end, ad.cos(x))
ad.sinh:define(function(x) return `C.sinh(x) end, ad.cosh(x))
ad.sqrt:define(function(x) return `C.sqrt(x) end, 1.0/(2.0*ad.sqrt(x)))
ad.tan:define(function(x) return `C.tan(x) end, 1.0 + ad.tan(x)*ad.tan(x))
ad.tanh:define(function(x) return `C.tanh(x) end, 1.0/(ad.cosh(x)*ad.cosh(x)))




function ad.select:propagatetype(args) return thallo_float, {bool,thallo_float,thallo_float} end
ad.select:define(function(x,y,z) 
    return quote
        var r : thallo_float
        if x then
            r = y
        else    
            r = z
        end
    in r end
end,0,x,ad.not_(x))

ad.abs:define(function(x) return `terralib.select(x >= 0,x,-x) end, ad.select(ad.greatereq(x, 0),1,-1))

function ad.and_(x,y) return x*y end

function ad.or_:propagatetype(args) return bool,{bool,bool} end
ad.or_:define(function(x,y) return `x or y end, 0, 0)

local comparisons = { "less", "greater", "lesseq", "greatereq", "eq", "neq" }
for i,c in ipairs(comparisons) do
    ad[c].propagatetype = function(self,args) return bool, {thallo_float,thallo_float} end
end 


ad.eq:define(function(x,y) return `x == y end, 0,0)
ad.neq:define(function(x,y) return `x ~= y end,0,0)
ad.less:define(function(x,y) return `x < y end, 0,0)
ad.greater:define(function(x,y) return `x > y end, 0,0)
ad.lesseq:define(function(x,y) return `x <= y end,0,0)
ad.greatereq:define(function(x,y) return `x >= y end,0,0)

function ad.not_:propagatetype(args) return bool, {bool} end
ad.not_:define(function(x) return `not x end, 0)
ad.identity:define(function(x) return x end,1) -- preserved across math optimizations

-- Use to treat the wrapped expression as a constant in autodiff
ad.constant:define(function(x) return `x end, 0)

setmetatable(ad,nil) -- remove special metatable that generates new blank ops

ad.Var,ad.Apply,ad.Const,ad.Exp = Var, Apply, Const, Exp
function ad.polysimplify(exps)
    if not use_polysimplify then return exps end
    local function sumtoterms(sum)
        assert(Apply:isclassof(sum) and sum.op.name == "sum")
        local terms = terralib.newlist()
        -- build internal list of terms
        for i,f in ipairs(sum:children()) do
            local c,ff = asprod(ad.polysimplify(f))
            local factor = {}
            for j,p in ipairs(ff) do
                local c,pp = aspowc(p)
                factor[pp] = c
            end
            if c ~= 1 then
                factor[toexp(c)] = 1
            end
            
            terms:insert(factor)
        end
        return terms,sum.const
    end

    local function createsum(terms,c)
        local factors = terralib.newlist()
        for i,t in ipairs(terms) do
            local pows = terralib.newlist()
            local keys = orderedexpressionkeys(t)
            for _,k in ipairs(keys) do
                local v = t[k]
                pows:insert(k^v)
            end
            factors:insert(ad.prod(1,unpack(pows)))
        end
        local r = ad.sum(assert(tonumber(c),"NaN?"),unpack(factors))
        return r
    end

    local function simplifylist(terms,c)

        -- short circuit when there is going to be no advantage to keep going
        if #terms == 0 then return toexp(c)
        elseif #terms == 1 then
            return createsum(terms,c)
        end

        local minpower,maxnegpower = {},{}
        local uses,neguses = {},{}

        -- count total uses
        for i,t in ipairs(terms) do
            for k,v in pairs(t) do
                local v = t[k]
                if v > 0 then
                    uses[k] = (uses[k] or 0) + 1
                    minpower[k] = math.min(minpower[k] or math.huge, v)
                else
                    neguses[k] = (neguses[k] or 0) + 1
                    maxnegpower[k] = math.max(maxnegpower[k] or -math.huge, v)
                end
            end
        end
        -- find maximum uses
        local maxuse,benefit,power,maxkey = 0,0
        local boolbonus = use_condition_factoring and 10 or 1
        local keys = orderedexpressionkeys(uses)
        for _,k in ipairs(keys) do
            local u = uses[k]
            local b = bool == k:type() and boolbonus*u or u
            if b > benefit then
                maxuse,maxkey,power,benefit = u,k,minpower[k],b
            end
        end
        local keys = orderedexpressionkeys(neguses)
        for _,k in ipairs(keys) do
            local u = neguses[k]
            local b = bool == k:type() and boolbonus*u or u
            if b > benefit then
                maxuse,maxkey,power,benefit = u,k,maxnegpower[k],b
            end
        end
        if maxuse < 2 then
            return createsum(terms,c) -- no benefit, so stop here
        end
        --partition terms
        local used,notused = terralib.newlist(),terralib.newlist()
        for i,t in ipairs(terms) do
            local v = t[maxkey]
            if v and ((v > 0 and power > 0) or (v < 0 and power < 0)) then
                local newv = v - power
                if newv == 0 then newv = nil end
                t[maxkey] = newv
                used:insert(t) 
            else
                notused:insert(t)
            end
        end

        -- simplify both sides, and make a new term

        --print("recurse",#notused,#used)
        local lhs = simplifylist(notused,0)
        local rhs = simplifylist(used,0)
        local r = ad.sum(c,lhs,maxkey^power * rhs)
        return r
    end
    
    local function dosimplify(exp)
        if Apply:isclassof(exp) then
            if exp.op.name == "sum" then
                return simplifylist(sumtoterms(exp))
            else
                local nargs = exp:children():map(ad.polysimplify)
                if exp.const then
                    return exp.op(exp.const,unpack(nargs))
                end
                local r = exp.op(unpack(nargs))
                return r
            end
        else 
            return exp
        end
    end
    return terralib.islist(exps) and exps:map(dosimplify) or dosimplify(exps)
end
-- generate two terms, one boolean-only term and one float only term
function ad.splitcondition(exp)
    if use_condition_factoring and Apply:isclassof(exp) and exp.op.name == "prod" then
        local cond,exp_ = one,one
        for i,e in ipairs(exp:children()) do
            if e:type() == bool then
                cond = cond * e
            else
                exp_ = exp_ * e
            end
        end
        return cond,exp_ * exp.const
    else
        return one,exp
    end
end

ad.newop = Op

return ad

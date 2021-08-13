local ad = require("ad")
local A = ad.classes

A:Extern("ExpLike",function(x) return ad.Exp:isclassof(x) or ad.ExpVector:isclassof(x) end)
A:Define [[
Dim = (string name, number size, number _index) unique
IndexSpace = (Dim* dims) unique
Offset = (IndexDomain* domains, number* data) unique

ImageType = (IndexSpace ispace, TerraType scalartype, number channelcount) unique
ImageLocation = ArgumentLocation(number idx) | UnknownLocation | StateLocation | JTJLocation
Image = (string name, ImageType type, boolean scalar, ImageLocation location)
# If resultdomain exists, then this is a contraction
PrecomputedDomain = (Image im, IterationDomain domain, IterationDomain? resultdomain)

Sparse = (string name, IndexSpace inspace, IndexSpace outspace)
ImageVector = (Image* images)
ProblemParam = ImageParam(ImageType imagetype, boolean isunknown)
             | ScalarParam(TerraType type)
             | SparseParam(IndexSpace inspace, IndexSpace outspace)
             attributes (string name, any idx)

IndexDomain = (Dim dim, string name, number index) unique

IndexComponent = DirectIndexComponent(IndexDomain domain) unique
        | SparseIndexComponent(SparseAccess access) unique
        | ConstantIndexComponent(number value) unique
        | ComponentBinOp(string op, IndexComponent lhs, IndexComponent rhs) unique

Index = ImageIndex(IndexComponent* components) unique
  | UnknownPairIndex(ImageIndex u0, ImageIndex u1) unique

SparseAccess = (Sparse sparse, Index index) unique

IndexAndSpacePair = (Index index, IndexSpace space) unique

IndexConstraint = (IndexComponent component, Dim dimension) unique

VarDef =  ImageAccess(Image image, Index index, number channel) unique
       | BoundsAccess(Offset min, Offset max) unique
       | IndexValue(IndexDomain indexdomain, number shift_) unique
       | ParamValue(string name,TerraType type) unique
       | SparseIndexValue(SparseAccess access, number index, number shift_) unique

FunctionKind = UnknownwiseFunction(IterationDomain domain) unique
             | IterationDomainwiseFunction(IterationDomain domain) #TODO make unique again
             | ResidualwiseFunction(IterationDomain domain, FunctionSchedule fnschedule) # Multiple residuals can be over the same domains!
             | ResidualAndContractionwiseFunction(IterationDomain domain, FunctionSchedule fnschedule) # Multiple residuals can be over the same domains!
             | GenericFunction(number numelements) unique

ResidualTemplate = (Exp expression, ImageAccess* unknowns)


KernelCostData = (number thread_count, number register_count, number opcount_per_thread, number memreads_per_thread, number memwrites_per_thread)

IterationDomain = (IndexDomain* domains) unique

# Tensor Contractions require this odd split
ResidualDomain = (IterationDomain full, IterationDomain external) unique

FunctionSchedule = AtOutputSchedule unique
                   | ResidualSchedule unique

JTJpSchedule = INLINE unique
            | PRECOMPUTE_JTJ(boolean sparse) unique
            | PRECOMPUTE_J(boolean sparse) unique
            | PRECOMPUTE_J_THEN_JTJ(boolean sparse) unique
            | APPLY_SEPARATELY unique


Schedule = (JTJpSchedule jtjpschedule, FunctionSchedule jtfschedule, FunctionSchedule fnschedule) unique

DomainAndSchedule = (ResidualDomain domain, Schedule schedule) unique

ResidualGroup = (DomainAndSchedule domainandschedule, ResidualTemplate* residuals)

NamedResidual = (string name, Exp* expressions)
Energy = (NamedResidual* residuals)

ScheduledEnergy = (ResidualGroup* residualgroups)

MaterializeInfo = (IndexDomain* domains)
JTFInfo = (boolean _compute_at_output)

ArrayArgument = UnknownArg
            | ResidualArg
            | UnknownPairArg
            attributes(string name)
FunctionSpec = (FunctionKind kind, string name, ArrayArgument* arguments, ExpLike* results, Scatter* scatters, ResidualGroup? derivedfrom)

Scatter = (Image image, Index index, number channel, Exp expression, string kind)
Condition = (IRNode* members) unique
IRNode = vectorload(ImageAccess value, number count) # another one
       | sampleimage(Image image, number count, IRNode* children)
       | reduce(string op, IRNode* children)
       | vectorconstruct(IRNode* children)
       | vectorextract(IRNode* children, number channel)
       | load(ImageAccess value)
       | intrinsic(VarDef value)
       | const(number value)
       | vardecl(number constant)
       | varuse(IRNode* children)
       | apply(string op, function generator, IRNode * children, number? const)
         attributes (TerraType type, Condition? condition)
ProblemSpec = ()
ProblemSpecAD = ()
SampledImage = (table op)
SampledImageArray = (table op)

TensorContraction = (IndexDomain* domains, Exp expression)
GradientImage = (ImageAccess* unknowns, Image image)
UnknownType = (ImageParam* images)

# Just for caching
ConditionIndexPair = (Condition condition, Index index) unique

JTJBlockType = (string name, ImageType imagetype) unique
JTJType = (JTJBlockType* blocks)

ResidualGroupType = (string name, ImageType imagetype) unique
ResidualType = (ResidualGroupType* groups)

ProblemFunctions = (FunctionKind typ, table functionmap)


ResidualGroupSchedule = (boolean jtj_materialize,
boolean j_materialize,
boolean jp_materialize, 
boolean compute_at_output, 
boolean jtf_compute_at_output,
IndexDomain* domain_order,
ExpLike* sumstoparallelize)

FullSchedule = (ResidualGroupSchedule* rgschedules, string* residualnames, ExpLike* exptoinline, ExpLike* exptoinlinegradient)

]]
return A
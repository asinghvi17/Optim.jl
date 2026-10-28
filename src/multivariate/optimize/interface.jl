default_options(method::AbstractOptimizer) = NamedTuple()

fallback_method(f) = NelderMead()
fallback_method(f, g!) = LBFGS()
fallback_method(f, g!, h!) = Newton()

# By default, use central finite difference method
const DEFAULT_AD_TYPE = ADTypes.AutoFiniteDiff(; fdtype = Val(:central))

function fallback_method(f::InplaceObjective)
    if !(f.fdf isa Nothing)
        if !(f.hvp isa Nothing)
            return KrylovTrustRegion()
        end
        return LBFGS()
    elseif !(f.fgh isa Nothing)
        return Newton()
    elseif !(f.fghvp isa Nothing)
        return KrylovTrustRegion()
    end
end

function fallback_method(f::NotInplaceObjective)
    if !(f.fdf isa Nothing)
        return LBFGS()
    elseif !(f.fgh isa Nothing)
        return LBFGS()
    else
        throw(
            ArgumentError(
                "optimize does not support $(typeof(f)) as the first positional argument",
            ),
        )
    end
end
fallback_method(f::NotInplaceObjective{<:Nothing,<:Nothing,<:Any}) = Newton()

fallback_method(d::OnceDifferentiable) = LBFGS()
fallback_method(d::TwiceDifferentiable) = Newton()

# `specialize=false` preserves interactive callable erasure; opt in with true
# for static compilation. Preconstructed objectives retain their chosen layout.
# promote the objective (tuple of callables or an AbstractObjective) according to method requirement
Base.@constprop :aggressive promote_objtype(method, x0, autodiff::ADTypes.AbstractADType, inplace::Bool, args...; specialize::Bool = false) =
    error("No default objective type for $method and $args.")
# actual promotions, notice that (args...) captures FirstOrderOptimizer and NonDifferentiable, etc
Base.@constprop :aggressive promote_objtype(method::ZerothOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, args...; specialize::Bool = false) =
    NonDifferentiable(args..., x, real(zero(eltype(x))); specialize)
Base.@constprop :aggressive promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f; specialize::Bool = false) =
    OnceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff, specialize)
Base.@constprop :aggressive promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, args...; specialize::Bool = false) =
    OnceDifferentiable(args..., x, real(zero(eltype(x))); inplace = inplace, specialize)
Base.@constprop :aggressive promote_objtype(method::FirstOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g, h; specialize::Bool = false) =
    OnceDifferentiable(f, g, x, real(zero(eltype(x))); inplace = inplace, specialize)
Base.@constprop :aggressive promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f; specialize::Bool = false) =
    TwiceDifferentiable(f, x, real(zero(eltype(x))); autodiff = autodiff, specialize)
Base.@constprop :aggressive promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    f::NotInplaceObjective;
    specialize::Bool = false) = TwiceDifferentiable(f, x, real(zero(eltype(x))); specialize)
Base.@constprop :aggressive promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    f::InplaceObjective;
    specialize::Bool = false) = TwiceDifferentiable(f, x, real(zero(eltype(x))); specialize)
Base.@constprop :aggressive promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g; specialize::Bool = false) =
    TwiceDifferentiable(
        f,
        g,
        x,
        real(zero(eltype(x)));
        inplace = inplace,
        autodiff = autodiff,
        specialize,
    )
Base.@constprop :aggressive promote_objtype(method::SecondOrderOptimizer, x, autodiff::ADTypes.AbstractADType, inplace::Bool, f, g, h; specialize::Bool = false) =
    TwiceDifferentiable(f, g, h, x, real(zero(eltype(x))); inplace = inplace, specialize)
# no-op
Base.@constprop :aggressive promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    nd::NonDifferentiable;
    specialize::Bool = false) = nd
Base.@constprop :aggressive promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    od::OnceDifferentiable;
    specialize::Bool = false) = od
Base.@constprop :aggressive promote_objtype(
    method::FirstOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    od::OnceDifferentiable;
    specialize::Bool = false) = od
Base.@constprop :aggressive promote_objtype(
    method::ZerothOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable;
    specialize::Bool = false) = td
Base.@constprop :aggressive promote_objtype(
    method::FirstOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable;
    specialize::Bool = false) = td
Base.@constprop :aggressive promote_objtype(
    method::SecondOrderOptimizer,
    x,
    autodiff::ADTypes.AbstractADType,
    inplace::Bool,
    td::TwiceDifferentiable;
    specialize::Bool = false) = td

# if no method or options are present
Base.@constprop :aggressive function optimize(
    f,
    x0::AbstractArray;
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    method = fallback_method(f)
    d = promote_objtype(method, x0, autodiff, inplace, f; specialize)

    options = Options(; default_options(method)...)
    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    x0::AbstractArray;
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
    inplace::Bool = true,
    specialize::Bool = false,
)

    method = fallback_method(f, g)

    d = promote_objtype(method, x0, autodiff, inplace, f, g; specialize)
 
    options = Options(; default_options(method)...)
    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    h,
    x0::AbstractArray;
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE, 
)
    method = fallback_method(f, g, h)
    d = promote_objtype(method, x0, autodiff, inplace, f, g, h; specialize)

    options = Options(; default_options(method)...)
    optimize(d, x0, method, options)
end

# no method supplied with objective
Base.@constprop :aggressive function optimize(
    d::T,
    x0::AbstractArray,
    options::Options,
) where {T<:AbstractObjective}
    optimize(d, x0, fallback_method(d), options)
end
# no method supplied with inplace and autodiff keywords because objective is not supplied
Base.@constprop :aggressive function optimize(
    f,
    x0::AbstractArray,
    options::Options;
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    method = fallback_method(f)
    d = promote_objtype(method, x0, autodiff, inplace, f; specialize)
    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    x0::AbstractArray,
    options::Options;
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)

    method = fallback_method(f, g)
    d = promote_objtype(method, x0, autodiff, inplace, f, g; specialize)
    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    h,
    x0::AbstractArray{T},
    options::Options;
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
) where {T}
    method = fallback_method(f, g, h)
    d = promote_objtype(method, x0, autodiff, inplace, f, g, h; specialize)

    optimize(d, x0, method, options)
end

# potentially everything is supplied (besides caches)
Base.@constprop :aggressive function optimize(
    f,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    d = promote_objtype(method, x0, autodiff, inplace, f; specialize)
    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    c::AbstractConstraints,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)

    d = promote_objtype(method, x0, autodiff, inplace, f; specialize)
    optimize(d, c, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
)
    d = promote_objtype(method, x0, autodiff, inplace, f, g; specialize)

    optimize(d, x0, method, options)
end
Base.@constprop :aggressive function optimize(
    f,
    g,
    h,
    x0::AbstractArray,
    method::AbstractOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
   
)
    d = promote_objtype(method, x0, autodiff, inplace, f, g, h; specialize)

    optimize(d, x0, method, options)
end

Base.@constprop :aggressive function optimize(
    d::D,
    x0::AbstractArray,
    method::SecondOrderOptimizer,
    options::Options = Options(; default_options(method)...);
    inplace::Bool = true,
    specialize::Bool = false,
    autodiff::ADTypes.AbstractADType = DEFAULT_AD_TYPE,
) where {D<:Union{NonDifferentiable,OnceDifferentiable}}
    d = promote_objtype(method, x0, autodiff, inplace, d; specialize)
    optimize(d, x0, method, options)
end

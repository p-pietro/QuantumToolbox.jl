export doqsdsolveProblem, doqsdsolveEnsembleProblem, doqsdsolve

#=
    Dynamically optimal quantum state diffusion (DO-QSD), see [Cao2025Dynamically](@cite).

    The optimal adaptive measurement phase of the channel `k` is

        u_k = i conj(C_k) / |C_k| ,      C_k = <O S_k>_ψ - <O>_ψ <S_k>_ψ ,

    which is, up to a sign (equivalent to dW_k -> -dW_k), the only unit phase with
    `Re(u_k C_k) = 0`, i.e. the one cancelling the direct Wiener noise of the target
    observable `O`.

    At `C_k = 0` every unit phase is optimal, and we take the deterministic fallback
    `u_k = 1`. A regularization such as `sqrt(abs2(C) + atol^2)` must not be used: it
    would give `|u_k| < 1`, which is a different (and not norm-preserving) unraveling.
=#
@inline function _doqsd_phase(C::T, atol::Real) where {T <: Complex}
    absC = abs(C)
    return absC > atol ? im * conj(C) / absC : one(T)
end

#=
    Per-channel coefficients of the DO-QSD unraveling, shared by the drift and the
    diffusion. Given `Sψ = Ŝ_k |ψ⟩`, `Oψ = Ô |ψ⟩`, `o = ⟨Ô⟩_ψ` and `n = ⟨ψ|ψ⟩`, return the
    optimal adaptive phase `u_k` and the real homodyne shift `x_k = Re(u_k ℓ_k)`.

    Every expectation value is a normalized ratio, which makes the drift and the diffusion
    homogeneous of degree one in `ψ`: the SDE then preserves `‖ψ‖` whatever its value, and
    the coefficients are insensitive to the norm drift of the integrator.
=#
@inline function _doqsd_coeffs(Sψ, ψ, Oψ, o, n, atol)
    ell = dot(ψ, Sψ) / n
    C = dot(Oψ, Sψ) / n - o * ell
    u = _doqsd_phase(C, atol)
    return u, real(u * ell)
end

#=
    Itô drift of the DO-QSD unraveling in the adaptive-homodyne gauge:

        A = [ -i Ĥ_eff + Σ_k ( x_k u_k Ŝ_k - x_k² / 2 ) ] |ψ⟩ ,

    where `-i Ĥ_eff = -i Ĥ - Σ_k Ŝ_k^† Ŝ_k / 2`, so the `-½ Ŝ_k^† Ŝ_k` term of the drift is
    already contained in `K`. Unlike the paper gauge, this drift depends on the adaptive
    phases, hence on the target observable `Ô`.

    Freezing the phases at `u_k = 1` gives `x_k = Re⟨Ŝ_k⟩_ψ = e_k/2` and reduces this to
    the drift integrated by `ssesolve`.
=#
struct DOQSDDriftOperator{KT, OpType <: Tuple, MT, CT <: AbstractVector, RT <: Real}
    K::KT
    sc_ops::OpType
    O::MT
    cache_O::CT
    cache_S::CT
    atol::RT
end

function (L::DOQSDDriftOperator)(du, u, p, t)
    n = real(dot(u, u))
    Oψ = L.cache_O
    Sψ = L.cache_S
    mul!(Oψ, L.O, u)
    o = real(dot(u, Oψ)) / n

    L.K(du, u, u, p, t) # du = -i * H_eff(t) * u; AbstractSciMLOperator in-place call is (w, v, u, p, t)

    # iterating over a homogeneous `Tuple` is type stable, which is the usual case here
    for op in L.sc_ops
        mul!(Sψ, op, u)
        uk, xk = _doqsd_coeffs(Sψ, u, Oψ, o, n, L.atol)
        @. du += (xk * uk) * Sψ - (xk^2 / 2) * u
    end
    return du
end

#=
    Itô diffusion of the DO-QSD unraveling in the adaptive-homodyne gauge:

        B_k = ( u_k Ŝ_k - x_k ) |ψ⟩ ,

    with the optimal adaptive phase `u_k` recomputed from the current (pre-increment)
    state only, so that the evaluation stays deterministic and side-effect free (adaptive
    solvers may reevaluate rejected stages).

    Column `k` of `w` is filled in place and is also used as the workspace for `Ŝ_k |ψ⟩`,
    so the only extra cache needed is `Ô |ψ⟩`. Freezing the phases at `u_k = 1` reduces
    this to `ssesolve`'s `M̂_k = Ŝ_k - e_k/2`.
=#
struct DOQSDDiffusionOperator{OpType <: Tuple, MT, CT <: AbstractVector, RT <: Real}
    sc_ops::OpType
    O::MT
    cache_O::CT
    atol::RT
end

function (L::DOQSDDiffusionOperator)(w, v, p, t)
    Nc = length(L.sc_ops)
    M = length(v)
    S = (size(w, 1), size(w, 2)) # supports also `w` as a `Vector`
    (S[1] == M && S[2] == Nc) || throw(DimensionMismatch("The size of the output matrix is incorrect."))

    n = real(dot(v, v))
    Oψ = L.cache_O
    mul!(Oψ, L.O, v)
    o = real(dot(v, Oψ)) / n

    for k in Base.OneTo(Nc)
        Bk = @view(w[:, k])
        mul!(Bk, L.sc_ops[k], v)
        uk, xk = _doqsd_coeffs(Bk, v, Oψ, o, n, L.atol)
        @. Bk = uk * Bk - xk * v
    end
    return w
end

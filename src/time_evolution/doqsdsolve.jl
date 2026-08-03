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

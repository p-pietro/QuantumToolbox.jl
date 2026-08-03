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

@doc raw"""
    doqsdsolveProblem(
        H::Union{AbstractQuantumObject{Operator},Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing,AbstractVector,Tuple,AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        e_ops::Union{Nothing,AbstractVector,Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        phase_atol::Union{Nothing,Real} = nothing,
        progress_bar::Union{Val,Bool} = Val(true),
        store_measurement::Union{Val,Bool} = Val(false),
        kwargs...,
    )

Generate the `SDEProblem` for the dynamically optimal quantum state diffusion (DO-QSD) unraveling of the Lindblad equation. This is defined by the following stochastic differential equation:

```math
d|\psi(t)\rangle = \left[-i \hat{H} + \sum_n \left(-\frac{1}{2} \hat{S}_n^\dagger \hat{S}_n - \frac{x_n^2}{2} + x_n u_n \hat{S}_n\right)\right] |\psi(t)\rangle dt + \sum_n \left(u_n \hat{S}_n - x_n\right) |\psi(t)\rangle dW_n(t)
```

where

```math
\ell_n = \langle \hat{S}_n \rangle_\psi,
\qquad
x_n = \mathrm{Re}\left(u_n \ell_n\right),
```

and the adaptive measurement phase ``u_n`` of each channel is chosen from the current state as

```math
u_n = i \frac{C_n^*}{|C_n|},
\qquad
C_n = \langle \hat{O} \hat{S}_n \rangle_\psi - \langle \hat{O} \rangle_\psi \langle \hat{S}_n \rangle_\psi,
```

with ``\hat{O}`` a Hermitian target observable (`op_target`). Since ``|u_n| = 1`` and ``\mathrm{Re}(u_n C_n) = 0``, the target observable loses its direct Wiener noise,

```math
d\langle \hat{O} \rangle_\psi = \langle \mathcal{L}^\dagger(\hat{O}) \rangle_\psi dt,
```

which makes the instantaneous growth rate of the trajectory variance of ``\langle \hat{O} \rangle_\psi`` minimal among all unravelings of this family. Whenever ``C_n = 0`` every unit phase is optimal, and the deterministic fallback ``u_n = 1`` is used. See [Cao2025Dynamically](@cite) for more details.

Above, ``\hat{S}_n`` are the stochastic collapse operators and ``dW_n(t)`` is the real Wiener increment associated to ``\hat{S}_n``. The equation is interpreted in the Itô sense.

# Arguments

- `H`: Hamiltonian of the system ``\hat{H}``. It can be either a [`QuantumObject`](@ref), a [`QuantumObjectEvolution`](@ref), or a `Tuple` of operator-function pairs.
- `ψ0`: Initial state of the system ``|\psi(0)\rangle``.
- `tlist`: List of time points at which to save either the state or the expectation values of the system.
- `sc_ops`: List of stochastic collapse operators ``\{\hat{S}_n\}_n``. It can be either a `Vector`, a `Tuple` or a [`AbstractQuantumObject`](@ref). It is recommended to use the last case when only one operator is provided. They must be time-independent.
- `op_target`: The Hermitian observable ``\hat{O}`` whose trajectory variance is minimized. It must be time-independent.
- `e_ops`: List of operators for which to calculate expectation values. It can be either a `Vector` or a `Tuple`.
- `params`: `NullParameters` of parameters to pass to the solver.
- `rng`: Random number generator for reproducibility.
- `phase_atol`: The adaptive phase falls back to ``u_n = 1`` when ``|C_n|`` is below this threshold. Defaults to `eps(T)^(3//4)`, with `T` the floating-point type of the problem. Passing `phase_atol = Inf` freezes every phase at ``u_n = 1``, which recovers [`ssesolve`](@ref).
- `progress_bar`: Whether to show the progress bar. Using non-`Val` types might lead to type instabilities.
- `store_measurement`: Not supported by DO-QSD, see the notes below.
- `kwargs`: The keyword arguments for the ODEProblem.

# Notes

- The states will be saved depend on the keyword argument `saveat` in `kwargs`.
- If `e_ops` is empty, the default value of `saveat=tlist` (saving the states corresponding to `tlist`), otherwise, `saveat=[tlist[end]]` (only save the final state). You can also specify `e_ops` and `saveat` separately.
- The default tolerances in `kwargs` are given as `reltol=2e-3` and `abstol=1e-3`.

!!! warning "Use fixed steps with more than one collapse operator"
    With non-diagonal noise the default algorithm chosen by [`doqsdsolve`](@ref) is `SRA2()`, which is formally an additive-noise method, and its adaptive step-size control frequently fails here. The adaptive phase rotates with ``\arg(C_n)``, so wherever ``|C_n|`` is small the diffusion becomes an almost discontinuous function of the state: the error estimate then stops shrinking with the step size and the integration aborts with `dt_min_unstable`. Measured on a driven cavity with two collapse operators, 15 of 20 trajectories abort, against roughly 1 in 20 for [`ssesolve`](@ref) on the same system; a two-channel qubit loses about 20% of its trajectories in *both* solvers. Integrate with fixed steps (`adaptive = false` together with a suitable `dt`) or supply an `alg` intended for non-diagonal multiplicative noise. Raising `phase_atol` also stabilizes the integration, but only at values large enough to suppress the adaptive phase itself, which defeats the purpose of this solver. A single collapse operator gives diagonal noise, uses `SRIW1()`, and is not affected.

- `store_measurement = Val(true)` is not supported: the DO-QSD measurement record ``dY_n = 2 \mathrm{Re}(u_n \ell_n) dt + dW_n`` depends on the adaptive phases, hence on the state, while the measurement callback only handles fixed operators. Use [`ssesolve`](@ref) for a fixed-phase homodyne record.
- The optimality is local in time: it minimizes the instantaneous growth of the variance of ``\langle \hat{O} \rangle_\psi``, and is not a proof of minimal variance at a prescribed final time. The observable can still acquire trajectory variance, because its drift depends on the stochastic state.
- For more details about `kwargs` please refer to [`DifferentialEquations.jl` (Keyword Arguments)](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)

!!! tip "Performance Tip"
    When `sc_ops` contains only a single operator, it is recommended to pass only that operator as the argument. This ensures that the stochastic noise is diagonal, making the simulation faster.

# Returns

- `prob`: The `SDEProblem` for the DO-QSD time evolution of the system.
"""
function doqsdsolveProblem(
        H::Union{AbstractQuantumObject{Operator}, Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing, AbstractVector, Tuple, AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        phase_atol::Union{Nothing, Real} = nothing,
        progress_bar::Union{Val, Bool} = Val(true),
        store_measurement::Union{Val, Bool} = Val(false),
        kwargs...,
    )
    haskey(kwargs, :save_idxs) &&
        throw(ArgumentError("The keyword argument \"save_idxs\" is not supported in QuantumToolbox."))

    getVal(makeVal(store_measurement)) && throw(
        ArgumentError(
            "The keyword argument \"store_measurement\" is not supported in doqsdsolve, since the DO-QSD measurement record depends on the state through the adaptive phases. Use ssesolve for a fixed-phase homodyne record.",
        ),
    )

    sc_ops isa Nothing &&
        throw(ArgumentError("The list of stochastic collapse operators must be provided. Use sesolveProblem instead."))
    sc_ops_list = _make_c_ops_list(sc_ops) # If it is an AbstractQuantumObject but we need to iterate
    sc_ops_isa_Qobj = sc_ops isa AbstractQuantumObject # We can avoid using non-diagonal noise if sc_ops is just an AbstractQuantumObject

    # the adaptive phases are evaluated from the raw operators, which are never updated in
    # time, so a time-dependent `sc_ops` would be silently frozen at its initial value
    all(isconstant, sc_ops_list) ||
        throw(ArgumentError("Time-dependent stochastic collapse operators are not supported in doqsdsolve."))
    ishermitian(op_target) || throw(ArgumentError("The target operator must be Hermitian."))

    H_eff_evo = _mcsolve_make_Heff_QobjEvo(H, sc_ops_list)
    isoper(H_eff_evo) || throw(ArgumentError("The Hamiltonian must be an Operator."))
    check_dimensions(H_eff_evo, op_target)

    # Convert initial state to dense vector with complex element type (T) and check dimensions
    T, ψ0, states_type, dimensions = _handle_init_state_and_sol_type_dims(H_eff_evo, ψ0)

    sc_ops_evo_data = Tuple(map(get_data ∘ QobjEvo, sc_ops_list))

    # `-i H_eff` already contains the `-½ Sₙ† Sₙ` term of the drift
    K = cache_operator(get_data(-1im * QuantumObjectEvolution(H_eff_evo)), ψ0)
    atol = isnothing(phase_atol) ? eps(_float_type(T))^(3 // 4) : _float_type(T)(phase_atol)

    O_data = get_data(op_target)
    A = DOQSDDriftOperator(K, sc_ops_evo_data, O_data, similar(ψ0), similar(ψ0), atol)
    B = DOQSDDiffusionOperator(sc_ops_evo_data, O_data, similar(ψ0), atol)

    tlist = _check_tlist(tlist, _float_type(T))

    kwargs2 = _merge_saveat(tlist, e_ops, default_sde_solver_options(T); kwargs...)
    kwargs3 = _merge_tstops(kwargs2, isconstant(H_eff_evo), tlist)
    kwargs4 = _generate_stochastic_kwargs(
        e_ops,
        sc_ops_list,
        makeVal(progress_bar),
        tlist,
        Val(false),
        kwargs3,
        SaveFuncSSESolve,
        T,
    )

    tspan = (tlist[1], tlist[end])
    noise = _make_noise(tspan[1], sc_ops, Val(false), rng)
    noise_rate_prototype = sc_ops_isa_Qobj ? nothing : similar(ψ0, length(ψ0), length(sc_ops_list))
    prob = SDEProblem{true}(
        A,
        B,
        ψ0,
        tspan,
        params;
        noise_rate_prototype = noise_rate_prototype,
        noise = noise,
        kwargs4...,
    )

    return TimeEvolutionProblem(prob, tlist, states_type, dimensions)
end

@doc raw"""
    doqsdsolveEnsembleProblem(
        H::Union{AbstractQuantumObject{Operator},Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing,AbstractVector,Tuple,AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        e_ops::Union{Nothing,AbstractVector,Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        prob_func::Union{Function,Nothing} = nothing,
        output_func::Union{Tuple,Nothing} = nothing,
        phase_atol::Union{Nothing,Real} = nothing,
        progress_bar::Union{Val,Bool} = Val(true),
        store_measurement::Union{Val,Bool} = Val(false),
        kwargs...,
    )

Generate the SDE `EnsembleProblem` for the dynamically optimal quantum state diffusion (DO-QSD) unraveling of the Lindblad equation. See [`doqsdsolveProblem`](@ref) for the definition of the stochastic differential equation and of the adaptive phases, and [Cao2025Dynamically](@cite) for more details.

# Arguments

- `H`: Hamiltonian of the system ``\hat{H}``. It can be either a [`QuantumObject`](@ref), a [`QuantumObjectEvolution`](@ref), or a `Tuple` of operator-function pairs.
- `ψ0`: Initial state of the system ``|\psi(0)\rangle``.
- `tlist`: List of time points at which to save either the state or the expectation values of the system.
- `sc_ops`: List of stochastic collapse operators ``\{\hat{S}_n\}_n``. It can be either a `Vector`, a `Tuple` or a [`AbstractQuantumObject`](@ref). It is recommended to use the last case when only one operator is provided. They must be time-independent.
- `op_target`: The Hermitian observable ``\hat{O}`` whose trajectory variance is minimized. It must be time-independent.
- `e_ops`: List of operators for which to calculate expectation values. It can be either a `Vector` or a `Tuple`.
- `params`: `NullParameters` of parameters to pass to the solver.
- `rng`: Random number generator for reproducibility.
- `ntraj`: Number of trajectories to use. Default is `500`.
- `ensemblealg`: Ensemble method to use. Default to `EnsembleThreads()`.
- `prob_func`: Function to use for generating the SDEProblem.
- `output_func`: a `Tuple` containing the `Function` to use for generating the output of a single trajectory, the (optional) `Progress` object, and the (optional) `RemoteChannel` object.
- `phase_atol`: The adaptive phase falls back to ``u_n = 1`` when ``|C_n|`` is below this threshold. Defaults to `eps(T)^(3//4)`, with `T` the floating-point type of the problem. Passing `phase_atol = Inf` freezes every phase at ``u_n = 1``, which recovers [`ssesolve`](@ref).
- `progress_bar`: Whether to show the progress bar. Using non-`Val` types might lead to type instabilities.
- `store_measurement`: Not supported by DO-QSD, see [`doqsdsolveProblem`](@ref).
- `kwargs`: The keyword arguments for the ODEProblem.

# Notes

- The states will be saved depend on the keyword argument `saveat` in `kwargs`.
- If `e_ops` is empty, the default value of `saveat=tlist` (saving the states corresponding to `tlist`), otherwise, `saveat=[tlist[end]]` (only save the final state). You can also specify `e_ops` and `saveat` separately.
- The default tolerances in `kwargs` are given as `reltol=2e-3` and `abstol=1e-3`.
- For more details about `kwargs` please refer to [`DifferentialEquations.jl` (Keyword Arguments)](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)

!!! tip "Performance Tip"
    When `sc_ops` contains only a single operator, it is recommended to pass only that operator as the argument. This ensures that the stochastic noise is diagonal, making the simulation faster.

# Returns

- `prob::EnsembleProblem with SDEProblem`: The Ensemble SDEProblem for the DO-QSD time evolution.
"""
function doqsdsolveEnsembleProblem(
        H::Union{AbstractQuantumObject{Operator}, Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing, AbstractVector, Tuple, AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        prob_func::Union{Function, Nothing} = nothing,
        output_func::Union{Tuple, Nothing} = nothing,
        phase_atol::Union{Nothing, Real} = nothing,
        progress_bar::Union{Val, Bool} = Val(true),
        store_measurement::Union{Val, Bool} = Val(false),
        kwargs...,
    )
    _prob_func =
        isnothing(prob_func) ?
        _ensemble_dispatch_prob_func(
            tlist,
            _stochastic_prob_func;
            sc_ops = sc_ops,
            store_measurement = Val(false),
        ) : prob_func
    _output_func =
        output_func isa Nothing ?
        _ensemble_dispatch_output_func(
            ensemblealg,
            progress_bar,
            ntraj,
            _standard_output_func;
            progr_desc = "[doqsdsolve] ",
        ) : output_func

    prob_doqsd = doqsdsolveProblem(
        H,
        ψ0,
        tlist,
        sc_ops,
        op_target;
        e_ops = e_ops,
        params = params,
        rng = rng,
        phase_atol = phase_atol,
        progress_bar = Val(false),
        store_measurement = makeVal(store_measurement),
        kwargs...,
    )

    ensemble_prob = TimeEvolutionProblem(
        EnsembleProblem(prob_doqsd, prob_func = _prob_func, output_func = _output_func[1], safetycopy = true),
        prob_doqsd.times,
        prob_doqsd.states_type,
        prob_doqsd.dimensions,
        (progr = _output_func[2], channel = _output_func[3], rng = rng),
    )

    return ensemble_prob
end

@doc raw"""
    doqsdsolve(
        H::Union{AbstractQuantumObject{Operator},Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing,AbstractVector,Tuple,AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        alg::Union{Nothing,AbstractSDEAlgorithm} = nothing,
        e_ops::Union{Nothing,AbstractVector,Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        prob_func::Union{Function,Nothing} = nothing,
        output_func::Union{Tuple,Nothing} = nothing,
        phase_atol::Union{Nothing,Real} = nothing,
        progress_bar::Union{Val,Bool} = Val(true),
        keep_runs_results::Union{Val,Bool} = Val(false),
        store_measurement::Union{Val,Bool} = Val(false),
        kwargs...,
    )

Dynamically optimal quantum state diffusion (DO-QSD) evolution of a quantum system, given the system Hamiltonian ``\hat{H}``, a list of stochastic collapse operators ``\{\hat{S}_n\}_n``, and a Hermitian target observable ``\hat{O}``:

```math
d|\psi(t)\rangle = \left[-i \hat{H} + \sum_n \left(-\frac{1}{2} \hat{S}_n^\dagger \hat{S}_n - \frac{x_n^2}{2} + x_n u_n \hat{S}_n\right)\right] |\psi(t)\rangle dt + \sum_n \left(u_n \hat{S}_n - x_n\right) |\psi(t)\rangle dW_n(t)
```

where ``\ell_n = \langle \hat{S}_n \rangle_\psi``, ``x_n = \mathrm{Re}(u_n \ell_n)``, and the measurement phase of each channel is adapted to the current state,

```math
u_n = i \frac{C_n^*}{|C_n|},
\qquad
C_n = \langle \hat{O} \hat{S}_n \rangle_\psi - \langle \hat{O} \rangle_\psi \langle \hat{S}_n \rangle_\psi .
```

Because ``\mathrm{Re}(u_n C_n) = 0``, the target observable follows ``d\langle \hat{O} \rangle_\psi = \langle \mathcal{L}^\dagger(\hat{O}) \rangle_\psi dt``: its direct Wiener noise is removed, and the instantaneous growth rate of its trajectory variance is minimal. This typically needs far fewer trajectories than [`ssesolve`](@ref) to resolve ``\langle \hat{O} \rangle`` at a given accuracy, while the trajectory average of any observable still reproduces [`mesolve`](@ref). See [Cao2025Dynamically](@cite) for more details.

# Arguments

- `H`: Hamiltonian of the system ``\hat{H}``. It can be either a [`QuantumObject`](@ref), a [`QuantumObjectEvolution`](@ref), or a `Tuple` of operator-function pairs.
- `ψ0`: Initial state of the system ``|\psi(0)\rangle``.
- `tlist`: List of time points at which to save either the state or the expectation values of the system.
- `sc_ops`: List of stochastic collapse operators ``\{\hat{S}_n\}_n``. It can be either a `Vector`, a `Tuple` or a [`AbstractQuantumObject`](@ref). It is recommended to use the last case when only one operator is provided. They must be time-independent.
- `op_target`: The Hermitian observable ``\hat{O}`` whose trajectory variance is minimized. It must be time-independent.
- `alg`: The algorithm to use for the stochastic differential equation. Default is `SRIW1()` if `sc_ops` is an [`AbstractQuantumObject`](@ref) (diagonal noise), and `SRA2()` otherwise (non-diagonal noise).
- `e_ops`: List of operators for which to calculate expectation values. It can be either a `Vector` or a `Tuple`.
- `params`: `NullParameters` of parameters to pass to the solver.
- `rng`: Random number generator for reproducibility.
- `ntraj`: Number of trajectories to use. Default is `500`.
- `ensemblealg`: Ensemble method to use. Default to `EnsembleThreads()`.
- `prob_func`: Function to use for generating the SDEProblem.
- `output_func`: a `Tuple` containing the `Function` to use for generating the output of a single trajectory, the (optional) `Progress` object, and the (optional) `RemoteChannel` object.
- `phase_atol`: The adaptive phase falls back to ``u_n = 1`` when ``|C_n|`` is below this threshold. Defaults to `eps(T)^(3//4)`, with `T` the floating-point type of the problem. Passing `phase_atol = Inf` freezes every phase at ``u_n = 1``, which recovers [`ssesolve`](@ref).
- `progress_bar`: Whether to show the progress bar. Using non-`Val` types might lead to type instabilities.
- `keep_runs_results`: Whether to save the results of each trajectory. Default to `Val(false)`. Use `Val(true)` together with [`std_expect`](@ref) to inspect the trajectory spread.
- `store_measurement`: Not supported by DO-QSD, see [`doqsdsolveProblem`](@ref).
- `kwargs`: The keyword arguments for the ODEProblem.

# Notes

- The states will be saved depend on the keyword argument `saveat` in `kwargs`.
- If `e_ops` is empty, the default value of `saveat=tlist` (saving the states corresponding to `tlist`), otherwise, `saveat=[tlist[end]]` (only save the final state). You can also specify `e_ops` and `saveat` separately.
- The default tolerances in `kwargs` are given as `reltol=2e-3` and `abstol=1e-3`.

!!! warning "Use fixed steps with more than one collapse operator"
    With non-diagonal noise (`sc_ops` given as a `Vector` or `Tuple`) the default algorithm is `SRA2()`, formally an additive-noise method, and its adaptive step-size control frequently fails here. The adaptive phase rotates with ``\arg(C_n)``, so wherever ``|C_n|`` is small the diffusion becomes an almost discontinuous function of the state: the error estimate then stops shrinking with the step size and the integration aborts with `dt_min_unstable`. Measured on a driven cavity with two collapse operators, 15 of 20 trajectories abort, against roughly 1 in 20 for [`ssesolve`](@ref) on the same system; a two-channel qubit loses about 20% of its trajectories in *both* solvers. Integrate with fixed steps (`adaptive = false` together with a suitable `dt`) or supply an `alg` intended for non-diagonal multiplicative noise. Raising `phase_atol` also stabilizes the integration, but only at values large enough to suppress the adaptive phase itself, which defeats the purpose of this solver. A single collapse operator gives diagonal noise, uses `SRIW1()`, and is not affected.

- The optimality is local in time, and only concerns `op_target`: other observables can be noisier than with [`ssesolve`](@ref).
- If ``\mathcal{L}^\dagger(\hat{O}) = \lambda \hat{O} + c \hat{I}``, then every single trajectory satisfies the deterministic equation ``d\langle \hat{O} \rangle_\psi = (\lambda \langle \hat{O} \rangle_\psi + c) dt``.
- For more details about `alg` please refer to [`DifferentialEquations.jl` (SDE Solvers)](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/)
- For more details about `kwargs` please refer to [`DifferentialEquations.jl` (Keyword Arguments)](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)

!!! tip "Performance Tip"
    When `sc_ops` contains only a single operator, it is recommended to pass only that operator as the argument. This ensures that the stochastic noise is diagonal, making the simulation faster.

# Returns

- `sol::TimeEvolutionStochasticSol`: The solution of the time evolution. See [`TimeEvolutionStochasticSol`](@ref).
"""
function doqsdsolve(
        H::Union{AbstractQuantumObject{Operator}, Tuple},
        ψ0::QuantumObject{Ket},
        tlist::AbstractVector,
        sc_ops::Union{Nothing, AbstractVector, Tuple, AbstractQuantumObject},
        op_target::QuantumObject{Operator};
        alg::Union{Nothing, AbstractSDEAlgorithm} = nothing,
        e_ops::Union{Nothing, AbstractVector, Tuple} = nothing,
        params = NullParameters(),
        rng::AbstractRNG = default_rng(),
        ntraj::Int = 500,
        ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
        prob_func::Union{Function, Nothing} = nothing,
        output_func::Union{Tuple, Nothing} = nothing,
        phase_atol::Union{Nothing, Real} = nothing,
        progress_bar::Union{Val, Bool} = Val(true),
        keep_runs_results::Union{Val, Bool} = Val(false),
        store_measurement::Union{Val, Bool} = Val(false),
        kwargs...,
    )
    ens_prob = doqsdsolveEnsembleProblem(
        H,
        ψ0,
        tlist,
        sc_ops,
        op_target;
        e_ops = e_ops,
        params = params,
        rng = rng,
        ntraj = ntraj,
        ensemblealg = ensemblealg,
        prob_func = prob_func,
        output_func = output_func,
        phase_atol = phase_atol,
        progress_bar = progress_bar,
        store_measurement = makeVal(store_measurement),
        kwargs...,
    )

    sc_ops_isa_Qobj = sc_ops isa AbstractQuantumObject # We can avoid using non-diagonal noise if sc_ops is just an AbstractQuantumObject

    if isnothing(alg)
        alg = sc_ops_isa_Qobj ? SRIW1() : SRA2()
    end

    return doqsdsolve(ens_prob, alg, ntraj, ensemblealg, makeVal(keep_runs_results))
end

# the solution is assembled exactly as for `ssesolve`, since both solvers store the
# expectation values with `SaveFuncSSESolve`
doqsdsolve(
    ens_prob::TimeEvolutionProblem,
    alg::AbstractSDEAlgorithm = SRA2(),
    ntraj::Int = 500,
    ensemblealg::EnsembleAlgorithm = EnsembleThreads(),
    keep_runs_results = Val(false),
) = ssesolve(ens_prob, alg, ntraj, ensemblealg, keep_runs_results)

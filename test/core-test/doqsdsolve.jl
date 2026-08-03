@testitem "doqsdsolve: optimal phase" begin
    # The optimal phase is the unit phase that cancels the direct Wiener noise of the
    # target observable, i.e. Re(u * C) = 0 with |u| = 1 (Theorem 4 of the paper).
    for C in (1.0 + 0.0im, -2.0 + 3.0im, 1.0e-3 * (0.3 - 0.7im), 1.0e5im)
        u = QuantumToolbox._doqsd_phase(C, 1.0e-12)
        @test abs(u) ≈ 1 atol = 1.0e-14
        @test real(u * C) ≈ 0 atol = 100 * eps(abs(C))
    end

    # At C = 0 every unit phase is optimal: use a fixed deterministic fallback.
    @test QuantumToolbox._doqsd_phase(0.0 + 0.0im, 1.0e-12) === 1.0 + 0.0im
    @test QuantumToolbox._doqsd_phase(1.0e-14 - 1.0e-15im, 1.0e-12) === 1.0 + 0.0im

    # The fallback must be a *unit* phase: no `sqrt(abs2(C) + atol^2)` regularization,
    # which would shrink |u| below 1 and change the unraveling.
    @test abs(QuantumToolbox._doqsd_phase(1.0e-11 + 0.0im, 1.0e-12)) ≈ 1 atol = 1.0e-14
end

@testitem "doqsdsolve: coefficient identities" begin
    using LinearAlgebra
    using Random

    rng = MersenneTwister(1234)

    N = 4
    a = destroy(N)
    H = 0.7 * a' * a + 0.3 * (a + a')
    sc_ops = [0.5 * a, 0.4 * (a + a'), 0.25 * a' * a]
    op_target = (a + a') / 2

    Hd = get_data(H)
    Od = get_data(op_target)
    Sd = map(get_data, sc_ops)

    # The coefficients are homogeneous of degree one in ψ, so every identity below must
    # hold for a deliberately unnormalized state too.
    ψ = randn(rng, ComplexF64, N)
    n = real(dot(ψ, ψ))
    o = real(dot(ψ, Od, ψ)) / n

    H_eff = H - 1im * sum(op -> op' * op, sc_ops) / 2
    K = cache_operator(get_data(QobjEvo(-1im * H_eff)), ψ) # `cache_operator` is exported
    drift = QuantumToolbox.DOQSDDriftOperator(K, Tuple(Sd), Od, similar(ψ), similar(ψ), 1.0e-12)
    diffusion = QuantumToolbox.DOQSDDiffusionOperator(Tuple(Sd), Od, similar(ψ), 1.0e-12)

    A = similar(ψ)
    B = similar(ψ, N, length(sc_ops))
    drift(A, ψ, nothing, 0.0)
    diffusion(B, ψ, nothing, 0.0)

    for (k, S) in enumerate(Sd)
        Bk = B[:, k]
        ell = dot(ψ, S, ψ) / n
        C = dot(ψ, Od * S, ψ) / n - o * ell
        u = QuantumToolbox._doqsd_phase(C, 1.0e-12)
        x = real(u * ell)

        # the phase is a unit phase cancelling the direct noise of the target observable
        @test abs(u) ≈ 1 atol = 1.0e-14
        @test real(u * C) ≈ 0 atol = 100 * eps(abs(C))
        # the diffusion column is exactly (u_k S_k - x_k) ψ
        @test Bk ≈ u * (S * ψ) - x * ψ
        # tangency of every diffusion column: 2 Re<ψ|B_k> = 0
        @test abs(real(dot(ψ, Bk))) < 1.0e-12 * n
        # direct observable-noise cancellation: <B_k|Oψ> + <ψ|O B_k> = 0
        @test abs(dot(Bk, Od, ψ) + dot(ψ, Od, Bk)) < 1.0e-12 * n
    end

    # Itô norm balance: 2 Re<ψ|A> + Σ_k <B_k|B_k> = 0, i.e. the SDE preserves ‖ψ‖
    @test 2 * real(dot(ψ, A)) + sum(abs2, B) ≈ 0 atol = 1.0e-10 * n

    # pointwise unraveling identity: A ψ' + ψ A' + Σ_k B_k B_k' = 𝓛(P) with P = |ψ><ψ|
    P = (ψ * ψ') / n
    LP = -1im * (Hd * P - P * Hd)
    for S in Sd
        global LP # `TestItemRunner` evaluates test items via `include_string`, which uses soft scope
        LP += S * P * S' - (S' * S * P + P * S' * S) / 2
    end
    @test (A * ψ' + ψ * A' + B * B') / n ≈ LP atol = 1.0e-10

    # the phase fallback must not blow up when C_k vanishes identically: for a Fock state
    # and O = a'a one has <O S> = <O><S> = 0 with S ∝ a, hence C = 0, u = 1 and x = 0
    diffusion_n = QuantumToolbox.DOQSDDiffusionOperator((Sd[1],), get_data(a' * a), similar(ψ), 1.0e-12)
    ψ_fock = get_data(fock(N, 2))
    Bn = similar(ψ_fock, N, 1)
    diffusion_n(Bn, ψ_fock, nothing, 0.0)
    @test all(isfinite, Bn)
    @test Bn[:, 1] ≈ Sd[1] * ψ_fock
end

@testitem "doqsdsolve: problem construction and argument checking" begin
    using LinearAlgebra

    N = 4
    a = destroy(N)
    H = a' * a
    ψ0 = fock(N, 1)
    tlist = range(0, 1, 11)
    sc_op = 0.5 * a
    op_target = a' * a

    prob = doqsdsolveProblem(H, ψ0, tlist, sc_op, op_target, progress_bar = Val(false))
    @test prob isa QuantumToolbox.TimeEvolutionProblem
    @test prob.prob.f.f isa QuantumToolbox.DOQSDDriftOperator
    @test prob.prob.g isa QuantumToolbox.DOQSDDiffusionOperator
    @test isnothing(prob.prob.noise_rate_prototype) # diagonal noise for a single operator
    @test prob.times == tlist

    # a `Vector` of operators gives non-diagonal noise
    prob2 = doqsdsolveProblem(H, ψ0, tlist, [sc_op, 0.3 * a' * a], op_target, progress_bar = Val(false))
    @test size(prob2.prob.noise_rate_prototype) == (N, 2)

    # argument checking
    @test_throws ArgumentError doqsdsolveProblem(H, ψ0, tlist, nothing, op_target, progress_bar = Val(false))
    @test_throws ArgumentError doqsdsolveProblem(H, ψ0, Float64[], sc_op, op_target, progress_bar = Val(false))
    @test_throws ArgumentError doqsdsolveProblem(H, ψ0, [0, 0.2, 0.1], sc_op, op_target, progress_bar = Val(false))
    @test_throws ArgumentError doqsdsolveProblem(H, ψ0, [0, 0.1, 0.1, 0.2], sc_op, op_target, progress_bar = Val(false))
    # the target observable must be Hermitian
    @test_throws ArgumentError doqsdsolveProblem(H, ψ0, tlist, sc_op, a, progress_bar = Val(false))
    # time-dependent stochastic collapse operators are not supported
    @test_throws ArgumentError doqsdsolveProblem(
        H, ψ0, tlist, QobjEvo((a, (p, t) -> exp(-t))), op_target, progress_bar = Val(false),
    )
    # the DO-QSD measurement record depends on the adaptive phases: not supported yet
    @test_throws ArgumentError doqsdsolveProblem(
        H, ψ0, tlist, sc_op, op_target, store_measurement = Val(true), progress_bar = Val(false),
    )
    @test_throws ArgumentError doqsdsolveProblem(
        H, ψ0, tlist, sc_op, op_target, save_idxs = [1], progress_bar = Val(false),
    )
    @test_throws DimensionMismatch doqsdsolveProblem(
        H, ψ0, tlist, sc_op, destroy(N + 1)' * destroy(N + 1), progress_bar = Val(false),
    )
end

@testitem "doqsdsolve: deterministic target observable" begin
    using LinearAlgebra
    using Random

    γ = 1.0
    ω = 2.0
    H = ω / 2 * sigmaz()
    sc_op = sqrt(γ) * sigmam()
    op_target = sigmaz()
    ψ0 = normalize(basis(2, 0) + basis(2, 1))
    tlist = range(0, 3 / γ, 31)
    ntraj = 8

    # 𝓛†(σz) = -γ (σz + 1) is affine in σz and i[H, σz] = 0, so under DO-QSD every
    # trajectory follows the deterministic law d<σz>/dt = -γ (<σz> + 1).
    z0 = real(expect(op_target, ψ0))
    analytic = @. -1 + (z0 + 1) * exp(-γ * tlist)

    # `qeye(2)` is used to divide out the (tiny) norm drift of the interpolated states
    e_ops = [op_target, qeye(2)]
    # 1.0e-6 is deliberate: it keeps the worst deviation from the analytic curve around
    # 1.0e-6, two orders below the assertions below, while 1.0e-8 makes this item take
    # minutes instead of seconds (adaptive SDE step count scales like tol^(-2/3))
    opts = (
        e_ops = e_ops, ntraj = ntraj, progress_bar = Val(false),
        keep_runs_results = Val(true), abstol = 1.0e-6, reltol = 1.0e-6,
    )

    sol = doqsdsolve(H, ψ0, tlist, sc_op, op_target; rng = MersenneTwister(42), opts...)
    sol_sse = ssesolve(H, ψ0, tlist, sc_op; rng = MersenneTwister(42), opts...)

    @test sol.ntraj == ntraj
    @test size(sol.expect) == (length(e_ops), ntraj, length(tlist))

    z = real.(sol.expect[1, :, :] ./ sol.expect[2, :, :])
    for j in Base.OneTo(ntraj)
        @test z[j, :] ≈ analytic atol = 1.0e-4
    end

    # zero trajectory variance for DO-QSD, large variance for the fixed-phase unraveling
    @test maximum(abs, std_expect(sol)[1, :]) < 1.0e-4
    @test maximum(abs, std_expect(sol_sse)[1, :]) > 0.05

    # the exact SDE preserves ‖ψ‖: monitor the numerical norm error
    @test maximum(abs, real.(sol.expect[2, :, :]) .- 1) < 1.0e-3

    # starting from |e> puts the first step exactly on the phase singularity C = 0, where
    # the deterministic fallback u = 1 is used: the deterministic law must still hold
    ψ_sing = basis(2, 0)
    tlist_sing = range(0, 1 / γ, 11)
    sol_sing = doqsdsolve(
        H, ψ_sing, tlist_sing, sc_op, op_target; e_ops = e_ops, ntraj = 4,
        rng = MersenneTwister(5), progress_bar = Val(false), keep_runs_results = Val(true),
        abstol = 1.0e-6, reltol = 1.0e-6
    )
    z_sing = real.(sol_sing.expect[1, :, :] ./ sol_sing.expect[2, :, :])
    analytic_sing = @. -1 + 2 * exp(-γ * tlist_sing)
    for j in Base.OneTo(4)
        @test z_sing[j, :] ≈ analytic_sing atol = 1.0e-3
    end

    # the show method and the averaged interface still work
    sol_avg = doqsdsolve(
        H, ψ0, tlist, sc_op, op_target; rng = MersenneTwister(42),
        e_ops = e_ops, ntraj = ntraj, progress_bar = Val(false)
    )
    @test size(sol_avg.expect) == (length(e_ops), length(tlist))
    @test real.(sol_avg.expect[1, :] ./ sol_avg.expect[2, :]) ≈ analytic atol = 1.0e-3
    @test isnothing(sol_avg.measurement)
    @test occursin("Solution of stochastic quantum trajectories", sprint((t, s) -> show(t, "text/plain", s), sol_avg))
end

@testitem "doqsdsolve: many-trajectory equivalence with mesolve" begin
    using LinearAlgebra
    using Random

    γ = 1.0
    nth = 0.2
    H = 0.5 * sigmaz()
    sc_ops = [sqrt(γ) * sigmam(), sqrt(γ * nth) * sigmap()]
    op_target = sigmaz()
    ψ0 = normalize(basis(2, 0) + basis(2, 1))
    tlist = range(0, 2 / γ, 21)
    e_ops = [op_target, sigmax(), qeye(2)]
    ntraj = 300

    sol_me = mesolve(H, ψ0, tlist, sc_ops, e_ops = e_ops, progress_bar = Val(false))
    # Fixed steps for both stochastic solvers: with the default adaptive `SRA2()` stepper,
    # a sizeable fraction of the 300 trajectories hit `dt_min_unstable` and abort early for
    # this two-channel system, biasing the ensemble average. `ssesolve` aborts at almost the
    # same rate under the same adaptive settings (measured 61/300 vs. 69/300 for doqsdsolve),
    # so this is a pre-existing adaptive-stepper stability limitation, not a doqsdsolve
    # coefficient bug; fixed steps avoid it entirely (0/300 aborts).
    stoch_opts = (alg = QuantumToolbox.SRA2(), dt = 1.0e-3, adaptive = false)
    sol = doqsdsolve(
        H, ψ0, tlist, sc_ops, op_target; e_ops = e_ops, ntraj = ntraj,
        rng = MersenneTwister(7), progress_bar = Val(false), keep_runs_results = Val(true), stoch_opts...
    )
    sol_sse = ssesolve(
        H, ψ0, tlist, sc_ops; e_ops = e_ops, ntraj = ntraj,
        rng = MersenneTwister(7), progress_bar = Val(false), keep_runs_results = Val(true), stoch_opts...
    )

    @test size(sol.expect) == (length(e_ops), ntraj, length(tlist))

    # the trajectory average reproduces the master equation for both observables
    avg = average_expect(sol)
    @test sum(abs, real(avg[1, :] .- sol_me.expect[1, :])) / length(tlist) < 0.02
    # measured 0.0346; 0.06 keeps CI headroom without weakening the check, since a broken unraveling would be off by ~1
    @test sum(abs, real(avg[2, :] .- sol_me.expect[2, :])) / length(tlist) < 0.06

    # 𝓛†(σz) is affine in σz also with two channels: <σz> stays deterministic
    z0 = real(expect(op_target, ψ0))
    z_ss = (nth - 1) / (1 + nth)
    analytic = @. z_ss + (z0 - z_ss) * exp(-γ * (1 + nth) * tlist)
    @test sum(abs, real(avg[1, :]) .- analytic) / length(tlist) < 0.02

    # variance reduction: essentially zero for the target, sizeable for the fixed phase
    std_doqsd = maximum(abs, std_expect(sol)[1, :])
    std_sse = maximum(abs, std_expect(sol_sse)[1, :])
    @test std_doqsd < 0.2 * std_sse
    @test std_sse > 0.05
end

@testitem "doqsdsolve: frozen phase reproduces ssesolve" begin
    using LinearAlgebra
    using Random

    # `phase_atol = Inf` makes `|C_n| > atol` false everywhere, so u_n ≡ 1. In the
    # adaptive-homodyne gauge that is exactly the fixed-phase unraveling integrated by
    # `ssesolve` (x_n = Re<S_n> = e_n/2), so both solvers must give the same trajectories.
    # Fixed steps AND matched `tstops` are both required for the two solvers to be driven
    # along the same Brownian path and thus be comparable trajectory by trajectory:
    # `ssesolve` always forces `tstops = tlist` (its `isconstant(K)` is never true, since K
    # carries state-dependent ScalarOperators), while `doqsdsolve` deliberately skips them
    # for a time-independent H. Mismatched `tstops` change where the noise process is
    # sampled, and `DiffEqNoiseProcess.jl`'s Brownian bridge fills in a different path:
    # measured max|Δexpect| = 4.2e-2 without matched tstops, 2.5e-8 with them, on this same
    # grid.
    N = 4
    a = destroy(N)
    H = a' * a + 0.3 * (a + a')
    sc_op = 0.7 * a
    op_target = a' * a
    ψ0 = fock(N, 1)
    tlist = range(0, 0.5, 6)
    e_ops = [op_target, a + a', qeye(N)]
    opts = (
        e_ops = e_ops, ntraj = 6, progress_bar = Val(false), keep_runs_results = Val(true),
        adaptive = false, dt = 1.0e-4, tstops = collect(tlist),
    )

    sol_frozen = doqsdsolve(
        H, ψ0, tlist, sc_op, op_target; phase_atol = Inf,
        rng = MersenneTwister(19), opts...
    )
    sol_sse = ssesolve(H, ψ0, tlist, sc_op; rng = MersenneTwister(19), opts...)

    # measured 2.5e-8; the residual comes from `ssesolve` calling `normalize!` on the
    # integrator state inside the coefficient update, where doqsdsolve uses norm-invariant
    # ratios instead.
    @test sol_frozen.expect ≈ sol_sse.expect atol = 1.0e-6

    # control: the agreement above is specific to the frozen phase, not because the two
    # solvers are trivially equal. With the adaptive phase, measured difference is 1.78.
    sol_adaptive = doqsdsolve(H, ψ0, tlist, sc_op, op_target; rng = MersenneTwister(19), opts...)
    @test maximum(abs, sol_adaptive.expect .- sol_sse.expect) > 0.1
end

@testitem "doqsdsolve: variance reduction for a non-closed observable" begin
    using LinearAlgebra
    using Random
    using Statistics

    N = 5
    a = destroy(N)
    γ = 1.0
    Δ = 0.5
    F = 0.8
    H = Δ * a' * a + F * (a + a')
    sc_op = sqrt(γ) * a
    op_target = a' * a
    ψ0 = fock(N, 0)
    tlist = range(0, 4 / γ, 21)
    e_ops = [op_target, qeye(N)]
    ntraj = 100

    # i[H, a'a] = i F (a - a') is not proportional to a'a, so <a'a> keeps a trajectory
    # spread through its drift: DO-QSD reduces the variance without cancelling it.
    sol_me = mesolve(H, ψ0, tlist, [sc_op], e_ops = e_ops, progress_bar = Val(false))
    opts = (e_ops = e_ops, ntraj = ntraj, progress_bar = Val(false), keep_runs_results = Val(true))
    sol = doqsdsolve(H, ψ0, tlist, sc_op, op_target; rng = MersenneTwister(11), opts...)
    sol_sse = ssesolve(H, ψ0, tlist, sc_op; rng = MersenneTwister(11), opts...)

    @test sum(abs, real(average_expect(sol)[1, :] .- sol_me.expect[1, :])) / length(tlist) < 0.05
    @test sum(abs, real(average_expect(sol_sse)[1, :] .- sol_me.expect[1, :])) / length(tlist) < 0.15

    std_doqsd = mean(abs, std_expect(sol)[1, :])
    std_sse = mean(abs, std_expect(sol_sse)[1, :])
    @test std_doqsd < 0.6 * std_sse
end

@testitem "doqsdsolve: type inference" begin
    using Random

    N = 4
    a = destroy(N)
    H = a' * a + 0.2 * (a + a')
    ψ0 = fock(N, 0)
    tlist = range(0, 1, 6)
    op_target = a' * a
    # a `Tuple` avoids the type instability of a `Vector` of operators
    sc_ops_tuple = (0.5 * a, 0.3 * (a + a'))
    e_ops = (op_target, a + a')
    rng = MersenneTwister(3)

    # Fixed steps are required here, and this is a real limitation worth understanding rather
    # than a test convenience. With more than one channel the default algorithm is `SRA2()`,
    # formally an additive-noise method, and DO-QSD's diffusion is far more state-sensitive
    # than `ssesolve`'s because the phase `u_k` rotates with `arg(C_k)`: wherever `|C_k|` is
    # small, an O(1)-magnitude diffusion column becomes an almost discontinuous function of
    # the state, so the adaptive error estimate does not shrink with `dt` and the step size
    # collapses to `dtmin`. Measured on this system with adaptive stepping: 15/20 trajectories
    # abort from the vacuum and 18/20 from (|0⟩+|1⟩)/√2, against ~1/20 for `ssesolve`.
    # Raising `phase_atol` only helps at values large enough to disable the adaptive phase
    # (1/20 aborts at 0.1), so fixed stepping is the remedy — see the docstring note.
    # This item tests type stability, not stepper robustness, so it pins the former only.
    fixed = (ntraj = 5, rng = rng, adaptive = false, dt = 1.0e-3)

    @inferred doqsdsolveEnsembleProblem(
        H, ψ0, tlist, sc_ops_tuple, op_target,
        e_ops = e_ops, progress_bar = Val(false), ntraj = 5, rng = rng,
    )
    @inferred doqsdsolve(H, ψ0, tlist, sc_ops_tuple, op_target; e_ops = e_ops, progress_bar = Val(false), fixed...)
    @inferred doqsdsolve(H, ψ0, tlist, sc_ops_tuple, op_target; progress_bar = Val(false), fixed...)
    @inferred doqsdsolve(H, ψ0, tlist, 0.5 * a, op_target; e_ops = e_ops, progress_bar = Val(false), fixed...)
    @inferred doqsdsolve(H, ψ0, tlist, sc_ops_tuple, op_target; e_ops = e_ops, progress_bar = Val(true), fixed...)
    # time-dependent Hamiltonian
    @inferred doqsdsolve(
        (H, (a + a', (p, t) -> 0.1 * cos(t))), ψ0, tlist, sc_ops_tuple, op_target;
        e_ops = e_ops, progress_bar = Val(false), fixed...,
    )
end

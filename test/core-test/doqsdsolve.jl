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

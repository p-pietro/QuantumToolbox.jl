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

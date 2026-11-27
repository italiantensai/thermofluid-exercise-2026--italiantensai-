# N07: 提供済みの独立した期待値による必須確認 → 受講生が編集する自作テスト。

module N07Tests
using Test, ThermofluidExercise
const N = ThermofluidExercise.N07
include("provided_support.jl")
include("analyze.jl")
# 配布済みの必須テスト。自作欄と区別し、入力・期待値を保持する。
@testset "N07 independent faces, half-cell walls and heat" begin
    a = Float64[1 2 4 3; 5 7 6 8; 9 10 12 11]
    saved = copy(a)
    b = similar(a)
    bc = N.channel_boundaries(:dirichlet)
    f = N.flux_buffers(a)
    @test N.thermal_stable_timestep(1., 0., 0.05, 0.5, 0.25, bc)≈0.8 / 5.0
    @test N.thermal_fluxes!(f, a, 0.5, 0.25; cx = 1., cy = 0., kappa = 0.05, bc)===f
    @test f.adv_x == [1 1 1 1; 1 2 4 3; 5 7 6 8; 9 10 12 11]
    @test f.diff_x≈[0 -0.2 -0.6 -0.4; -0.4 -0.5 -0.2 -0.5; -0.4 -0.3 -0.6 -0.3; 0 0 0 0]
    @test f.adv_y == zeros(3, 5)
    @test f.diff_y≈[-0.4 -0.2 -0.4 0.2 1.2; -2 -0.4 0.2 -0.4 3.2; -3.6 -0.2 -0.4 0.2 4.4]
    r = N.thermal_step!(b, a, 0.01, 0.5, 0.25; cx = 1., cy = 0., kappa = 0.05, bc)
    @test r.temperature===b
    @test collect(r.boundary_rates.advective) == [1., -10.5, 0., 0.]
    @test collect(r.boundary_rates.diffusive)≈[-0.3, 0., -3., -4.4]
    @test b≈[1.0 1.994 3.908 2.922; 4.856 6.872 5.992 7.752; 8.776 9.942 11.844 10.766] atol =
        1e-14
    @test 0.5 * 0.25 * sum(b - a)≈0.01 * (-17.2) atol = 1e-14
    @test a == saved
    for wall in (:dirichlet, :insulated)
        bc = N.channel_boundaries(wall)
        q0 = 0.125sum(a)
        total = 0.
        u = copy(a)
        v = similar(u)
        for _ in 1:20
            r = N.thermal_step!(v, u, 0.01, 0.5, 0.25; cx = 1., cy = 0., kappa = 0.05, bc)
            rates = r.boundary_rates
            total+=0.01 * (sum(rates.advective) + sum(rates.diffusive))
            u, v = v, u
        end
        @test 0.125sum(u) - q0≈total atol = 1e-12
        wall == :insulated &&
            @test r.boundary_rates.diffusive.south == r.boundary_rates.diffusive.north == 0.
    end
    for kind in (:periodic, :insulated, :dirichlet)
        bc = N.boundaries(kind; value = kind == :dirichlet ? 2. : 0.)
        N.thermal_step!(
            b,
            fill(2., 3, 4),
            0.01,
            0.5,
            0.25;
            cx = 0.,
            cy = 0.,
            kappa = 0.05,
            bc,
        )
        @test b≈fill(2., 3, 4)
    end
    bc = N.boundaries(:dirichlet)
    bound = 1 / (0.05 * (3 / 0.5 ^ 2 + 3 / 0.25 ^ 2))
    impulse = zeros(3, 4)
    impulse[1, 1] = 1.
    N.thermal_step!(b, impulse, bound, 0.5, 0.25; cx = 0., cy = 0., kappa = 0.05, bc)
    @test minimum(b) >= -1e-15
    fill!(b, -99.)
    @test_throws ArgumentError N.thermal_step!(
        b,
        a,
        bound * (1 + 1e-8),
        0.5,
        0.25;
        cx = 0.,
        cy = 0.,
        kappa = 0.05,
        bc,
    )
    @test all(==(-99.), b)
    @test_throws ArgumentError N.thermal_step!(
        b,
        a,
        1 / (0.1 * (1 / 0.5 ^ 2 + 1 / 0.25 ^ 2)),
        0.5,
        0.25;
        cx = 0.,
        cy = 0.,
        kappa = 0.05,
        bc,
    )
    # Signed periodic flow at both ends, old-state use and heat cancellation.
    bc = N.boundaries(:periodic)
    for (cx, cy, kappa) in ((-1., 0.5, 0.), (0., 0., 0.05), (0., 0., 0.))
        r = N.thermal_step!(b, a, 0.01, 0.5, 0.25; cx, cy, kappa, bc)
        @test sum(b)≈sum(a) atol = 1e-12
        @test sum(r.boundary_rates.advective) + sum(r.boundary_rates.diffusive)≈0. atol =
            1e-13
        if kappa == 0 && cx != 0
            @test b[3, 1]≈8.88
        elseif cx == cy == kappa == 0
            @test b == a
        end
    end
end
@testset "N07 two-component signed advective Burgers" begin
    u = [1. -2. 0. 0.5; -0.5 2. -1. 0.; 3. -1. 0.2 -2.]
    v = [-0.5 1. 2. -1.; 2. -1. 0. 0.5; 0. 0.5 -2. 1.]
    originals = (copy(u), copy(v))
    un = similar(u)
    vn = similar(v)
    dt = 0.001
    dx = 0.5
    dy = 0.25
    nu = 0.05
    r = N.burgers_step!(un, vn, u, v, dt, dx, dy; nu)
    @test r.u===un && r.v===vn
    @test un≈[
        0.9953 -1.967 -0.01736 0.4989;
        -0.4891 1.9662 -0.99396 -0.0035;
        2.9707 -0.98944 0.17892 -1.97454
    ] atol = 1e-14
    @test vn≈[
        -0.4946 0.9851 1.9876 -0.9925;
        1.9815 -0.9841 -0.0044 0.4996;
        0.0135 0.4974 -1.9696 0.9763
    ] atol = 1e-14
    @test u == originals[1] && v == originals[2]
    @test N.burgers_stable_timestep(u, v, nu, dx, dy)≈0.8 / (9.0 + 2.0)
    @test N.burgers_stable_timestep(2u, 2v, nu, dx, dy) <
          N.burgers_stable_timestep(u, v, nu, dx, dy)
    a = fill(2., 3, 4)
    b = fill(-1., 3, 4)
    N.burgers_step!(un, vn, a, b, 0.001, dx, dy; nu = 0.)
    @test un == a && vn == b
    N.burgers_step!(un, vn, u, zeros(3, 4), dt, dx, dy; nu = 0.)
    @test all(iszero, vn)
    arrays = [copy(u) for _ in 1:4]
    for (i, j) in ((1, 3), (1, 2))
        args = copy(arrays)
        args[j] = args[i]
        @test_throws ArgumentError N.burgers_step!(args..., dt, dx, dy; nu)
    end
    fill!(un, -99.)
    fill!(vn, -88.)
    @test_throws ArgumentError N.burgers_step!(un, vn, u, v, 1., dx, dy; nu)
    @test all(==(-99.), un) && all(==(-88.), vn)
    @test_throws ArgumentError N.burgers_stable_timestep(u, v, -1., dx, dy)
end

@testset "N07 配布済み熱収支解析" begin
    b = N07Analysis.heat_budget(
        [0., 0.25, 1.],
        [2., 4., 5.],
        [3. 4.; -1. -2.; 0. 0.; 0. 0.],
        [0.5 0.2; 0. 0.; -0.2 -0.4; -0.3 -0.8],
    )
    @test b.net_input≈[2., 1.]
    @test b.cumulative_input≈[0., 2., 3.]
    @test b.residual≈zeros(3) atol = 1e-14
    @test_throws ArgumentError N07Analysis.heat_budget(
        [0., 0.],
        [1., 1.],
        zeros(4, 1),
        zeros(4, 1),
    )
end
@testset "N07 自作: 開境界2条件の辺別熱収支" begin
    # 独立なセル中心・ゴーストセルのPython計算による12ステップの期待値。
    # dt=0.01, dx=0.5, dy=0.25, cx=1, kappa=0.05。積算の丸めにatol=1e-12。
    initial = [1. 2. 4. 3.; 5. 7. 6. 8.; 9. 10. 12. 11.]
    expected_fields = (
        [0.982122928441346 1.89090881935051 3.080158083317713 2.2218034806589215;
         3.5857740247008665 5.673091659789037 5.646319513321143 5.575266088505036;
         6.70818018903313 9.120544856576116 10.310208813547609 8.465654319580253],
        [1.1491451843519758 1.898387614537551 3.0990128613600296 2.64896052398954;
         4.307781470911163 5.706392527351002 5.6987482749629335 6.73257336953615;
         8.073554840951438 9.182956536790478 10.388171534040008 10.194739971491044],
    )
    expected_adv = ([0.12, -1.1547597927181195, 0., 0.],
                    [0.12, -1.2024049214166939, 0., 0.])
    expected_diff = ([-0.03068476985742098, 0., -0.3162130967994221, -0.46083824352232744],
                     [-0.032541989799143044, 0., 0., 0.])
    for (case, wall) in enumerate((:dirichlet, :insulated))
        bc = N.channel_boundaries(wall)
        old, new = copy(initial), similar(initial)
        adv, diff = zeros(4), zeros(4)
        for step in 1:12
            snapshot = copy(old)
            r = N.thermal_step!(new, old, 0.01, 0.5, 0.25; cx = 1., cy = 0., kappa = 0.05, bc)
            @test old == snapshot
            adv .+= 0.01 .* collect(values(r.boundary_rates.advective))
            diff .+= 0.01 .* collect(values(r.boundary_rates.diffusive))
            @test 0.125 * (sum(new) - sum(initial)) ≈ sum(adv) + sum(diff) atol = 1e-12 rtol = 0
            if wall == :insulated
                @test r.boundary_rates.diffusive.south == r.boundary_rates.diffusive.north == 0
            end
            old, new = new, old
        end
        @test old ≈ expected_fields[case] atol = 1e-12 rtol = 0
        @test adv ≈ expected_adv[case] atol = 1e-12 rtol = 0
        @test diff ≈ expected_diff[case] atol = 1e-12 rtol = 0
    end
end
include("simulate.jl")
@testset "N07 自作: 温度とBurgersの解析解収束" begin
    # 公式3格子をメモリ内で再計算し，保存結果に依存せず解析解と比較する。
    for id in ("periodic_diffusion", "periodic_combined", "periodic_burgers")
        errors = Float64[]
        component_errors = (Float64[], Float64[])
        for (nx, ny) in OFFICIAL_GRIDS
            c = N07Simulation.simulate_case(id, nx, ny)
            if id == "periodic_burgers"
                ue, ve = exact_burgers(c["x"], c["y"], 1.)
                eu = sqrt(sum(abs2, c["u"][:, :, end] - ue) / (nx * ny))
                ev = sqrt(sum(abs2, c["v"][:, :, end] - ve) / (nx * ny))
                push!(component_errors[1], eu)
                push!(component_errors[2], ev)
                push!(errors, hypot(eu, ev))
            else
                exact = exact_temperature(c["x"], c["y"], 1., id)
                push!(errors, sqrt(sum(abs2, c["temperature"][:, :, end] - exact) / (nx * ny)))
            end
        end
        lo, hi = id == "periodic_diffusion" ? (1.8, 2.2) : (0.8, 1.2)
        @test all(isfinite, errors) && all(errors .> 0)
        @test all(diff(errors) .< 0)
        @test all(lo .<= log2.(errors[1:2] ./ errors[2:3]) .<= hi)
        if id == "periodic_burgers"
            for e in component_errors
                @test all(diff(e) .< 0)
                @test all(0.8 .<= log2.(e[1:2] ./ e[2:3]) .<= 1.2)
            end
        end
    end
end
@testset "N07 保存結果と来歴" begin
    @test check_complete()
end
end

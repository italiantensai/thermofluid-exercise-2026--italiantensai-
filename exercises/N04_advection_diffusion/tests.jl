# 必須テストは提供済みの期待値で確認します。末尾の自作テストだけを受講生が編集します。
# 入力の準備 → 独立した期待値との比較 → 入力保持・不正入力の確認の順に読みます。

using Test
if !isdefined(Main, :N04AdvectionDiffusion)
    include(joinpath(@__DIR__, "run.jl"))
end
import .N04AdvectionDiffusion
const N04 = N04AdvectionDiffusion
N04.validate_model(N04.SELECTED_MODEL)
@testset "N04 必須テスト" begin
    model = N04.SELECTED_MODEL
    @testset "非対称1ステップ・周期端・旧配列不変" begin
        old = [1., 2., 4., 3.]
        saved = copy(old)
        new = similar(old)
        expected = model == :linear ? [1.26, 1.92, 3.74, 3.08] : [1.46, 1.87, 3.34, 3.33]
        @test N04.advection_diffusion_step!(new, old, 0.1, 1., 0.2) === new
        @test new ≈ expected atol = 1e-14
        @test old == saved
    end
    @testset "定数場と周期積分" begin
        old = fill(1.7, 5)
        new = similar(old)
        N04.advection_diffusion_step!(new, old, 0.1, 1., 0.2)
        @test new ≈ old
        @test N04.conserved_integral([1., 2., 4., 3.], 0.5) == 5.
    end
    @testset "移流・拡散をゼロにした極限" begin
        old = [1., 2., 4., 3.]
        new = similar(old)
        N04.advection_diffusion_step!(new, old, 0.1, 1., 0.2; advection = false)
        @test new ≈ [1.06, 2.02, 3.94, 2.98]
        N04.advection_diffusion_step!(new, old, 0.1, 1., 0.)
        @test new ≈ (model == :linear ? [1.2, 1.9, 3.8, 3.1] : [1.4, 1.85, 3.4, 3.35])
    end
    @testset "合成刻み・最終時刻・標準安定性" begin
        @test N04.stable_timestep(2., 0.5, 0.1) ≈ 1 / 6
        r = N04.simulate()
        @test r.steps * r.dt ≈ r.t_final == 1.
        @test r.max_stability_number <= 0.8 + 1e-12
        @test all(isfinite, r.u) && 1 - 1e-12 <= r.minimum <= r.maximum <= 2 + 1e-12
    end
end
@testset "N04 自作テスト" begin
    # TODO(自作): 非対称な非定数配列を複数ステップ進め、周期積分の保存を確認する。
    # 初期積分を独立に計算し、点数・ステップ数・値の大きさから許容誤差を説明する。
    @testset "非対称周期配列の60ステップ積分保存" begin
        initial = [1.0, 1.8, 1.1, 2.0, 1.4, 0.7]
        old = copy(initial)
        new = similar(old)
        dx = 0.25
        steps = 60
        # 独立した手計算: 0.25*(1+1.8+1.1+2+1.4+0.7)=2.0。
        expected_integral = 2.0
        # C=0.16、Fo=0.032、C+2Fo=0.224で合成条件を満たす。
        for _ in 1:steps
            N04.advection_diffusion_step!(new, old, 0.04, dx, 0.05)
            old, new = new, old
        end
        tolerance = 10eps(Float64) * length(initial) * steps * maximum(abs, initial) * dx
        @test isapprox(N04.conserved_integral(old, dx), expected_integral;
            atol=tolerance, rtol=0)
    end
    # TODO(自作): 選択モデルの解析解と40・80・160点の誤差・観測次数を確認する。
    # 誤差の指標、同じ最終時刻、0.8〜1.2の次数範囲、検出できない誤りも説明する。
    @testset "線形周期解析モードの約1次の格子収束" begin
        errors = Float64[]
        for nx in (40, 80, 160)
            r = N04.simulate(; nx, initial=:mode, diffusivity=0.1,
                speed=1.0, safety=0.8, t_final=1.0)
            # 提供解析解を呼ばず、線形モデルの期待解をテスト側に明示。
            exact = 1 .+ 0.5 * exp(-0.1 * pi^2) .* sin.(pi .* (r.x .- 1))
            push!(errors, maximum(abs.(r.u - exact)))
        end
        orders = log2.(errors[1:2] ./ errors[2:3])
        @test errors[1] > errors[2] > errors[3] > 0
        @test all(p -> 0.8 < p < 1.2, orders)
    end
end

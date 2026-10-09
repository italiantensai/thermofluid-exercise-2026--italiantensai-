using Test

if !isdefined(Main, :N01LinearAdvection)
    include(joinpath(@__DIR__, "run.jl"))
end

@testset "N01 必須テスト" begin
    # TODO(必須): 座標、base、plateau、区間端を自分で選び、rectangular_initial_conditionが返す配列全体を手計算した期待値と比較する。
    @testset "矩形の端点を含む初期条件" begin
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        initial = N01LinearAdvection.rectangular_initial_condition(
            x; base = 1.0, plateau = 2.0, plateau_start = 0.5, plateau_end = 1.0,
        )
        @test initial == [1.0, 2.0, 2.0, 1.0, 1.0]
        @test x == [0.0, 0.5, 1.0, 1.5, 2.0]
    end

    # TODO(必須): 小さなu_old、移流速度、時間刻み、格子幅を選び、upwind_step!後のu_new配列全体を手計算した期待値と比較する。
    @testset "風上差分の1段階更新" begin
        u_old = [1.0, 2.0, 4.0, 3.0, 1.0]
        u_new = zeros(5)
        # c*dt/dx = 0.5。内部は左隣と現在値の平均になる。
        returned = N01LinearAdvection.upwind_step!(u_new, u_old, 1.0, 0.25, 0.5)
        @test u_new == [1.0, 1.5, 3.0, 3.5, 1.0]
        @test u_old == [1.0, 2.0, 4.0, 3.0, 1.0]
        @test returned === u_new
    end

    # TODO(必須): 同じ入力でcentered_step!後のu_new配列全体を手計算した期待値と比較する。風上差分との違いが現れる入力を選ぶ。
    @testset "中心差分の1段階更新" begin
        u_old = [1.0, 2.0, 4.0, 3.0, 1.0]
        u_new = zeros(5)
        # c*dt/(2dx) = 0.25。内部は2-0.25*3、4-0.25*1、3-0.25*(-3)。
        returned = N01LinearAdvection.centered_step!(u_new, u_old, 1.0, 0.25, 0.5)
        @test u_new == [1.0, 1.25, 3.75, 3.75, 1.0]
        @test u_old == [1.0, 2.0, 4.0, 3.0, 1.0]
        @test returned === u_new
    end

    # TODO(必須): apply_boundary!が左端と右端を課題の境界条件どおりに書き換えることを、配列全体の期待値で確かめる。
    @testset "固定流入とゼロ勾配流出" begin
        u = [9.0, 2.0, 4.0, 3.0, 8.0]
        returned = N01LinearAdvection.apply_boundary!(u; left_value = 1.0)
        @test u == [1.0, 2.0, 4.0, 3.0, 3.0]
        @test returned === u
    end

    # TODO(必須): simulateで安定な風上差分と不安定な中心差分を同じ条件で計算し、最終値の範囲または振幅から両者の違いを確かめる。条件と判定基準は自分で書く。
    @testset "同条件での有界性と振動" begin
        upwind = N01LinearAdvection.simulate(
            ; scheme = :upwind, nx = 81, c = 1.0, cfl = 0.5, t_final = 0.5,
        )
        centered = N01LinearAdvection.simulate(
            ; scheme = :centered, nx = 81, c = 1.0, cfl = 0.5, t_final = 0.5,
        )
        # 数値の丸めだけを許す風上法の範囲検査と、1e-3を超える範囲逸脱の検出。
        tolerance = 100eps(Float64)
        @test upwind.minimum >= 1.0 - tolerance
        @test upwind.maximum <= 2.0 + tolerance
        @test centered.minimum < 1.0 - 1e-3 || centered.maximum > 2.0 + 1e-3
        @test upwind.cfl ≈ 0.5
        @test upwind.dt * upwind.steps ≈ 0.5
        @test centered.dt == upwind.dt && centered.steps == upwind.steps
    end
end

@testset "N01 自作テスト" begin
    # TODO(自作): 必須とは異なる初期条件、CFL、格子数、またはAPIの性質を一つ選び、期待する結果を自分で書く。PNG、TOML、ファイルサイズは対象にしない。
    @testset "CFL=1で内部の分布が1格子分移る" begin
        # 必須のCFL=0.5とは異なり、CFL=1では左隣の値がそのまま内部へ移る。
        u_old = [1.0, 4.0, 2.0, 3.0, 1.0]
        u_new = zeros(5)
        N01LinearAdvection.upwind_step!(u_new, u_old, 2.0, 0.25, 0.5)
        @test u_new == [1.0, 1.0, 4.0, 2.0, 1.0]
        @test u_old == [1.0, 4.0, 2.0, 3.0, 1.0]
    end
end

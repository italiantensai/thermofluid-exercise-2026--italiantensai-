# N05の共通API。周期添字 → 流束 → バッファ検証 → 刻み調整 → 既習課題の入口の順に読みます。
# N01〜N04の更新と境界処理は課題別のサブモジュールに置きます。

module ThermofluidExercise

function validate_periodic_index(i, n)
    i isa Integer && n isa Integer && !(i isa Bool || n isa Bool) &&
        n >= 1 && 1 <= i <= n ||
        throw(ArgumentError("添字は1 <= i <= n、点数はn >= 1の整数です（Bool不可）"))
    return nothing
end

finite_real(value, name) =
    value isa Real && isfinite(value) ? nothing :
    throw(ArgumentError("$name は有限な実数です"))

function timestep_count(count)
    isfinite(count) && count <= typemax(Int) ||
        throw(ArgumentError("ステップ数が表現範囲を超えています"))
    return max(1, ceil(Int, count))
end

"""
    periodic_left_index(i, n)

1始まりの周期配列で左隣の添字を求める。

# 引数

- `i`: 1始まりの現在の添字。
- `n`: 配列の点数。

i,nはBoolを除く整数、n>=1、1<=i<=n。

# 返り値

1からnまでの整数。
"""
function periodic_left_index(i, n)
    validate_periodic_index(i, n)
    return i == 1 ? n : i - 1
end

"""
    periodic_right_index(i, n)

1始まりの周期配列で右隣の添字を求める。

# 引数

- `i`: 1始まりの現在の添字。
- `n`: 配列の点数。

i,nはBoolを除く整数、n>=1、1<=i<=n。

# 返り値

1からnまでの整数。
"""
function periodic_right_index(i, n)
    validate_periodic_index(i, n)
    return i == n ? 1 : i + 1
end

"""
    linear_flux(u, speed)

有限な実数の線形流束を評価する。

# 引数

- `u`: 有限な実数の場の値。
- `speed`: 有限な実数の移流速度。

u,speedの有限実数を検証する。

# 返り値

スカラーの線形流束。
"""
function linear_flux(u, speed)
    finite_real(u, "u")
    finite_real(speed, "speed")
    return speed * u
end

"""
    burgers_flux(u)

有限な実数のBurgers流束を評価する。

# 引数

- `u`: 有限な実数の場の値。

負値も受け付け、有限実数を検証する。

# 返り値

スカラーのBurgers流束。
"""
function burgers_flux(u)
    finite_real(u, "u")
    return u^2 / 2
end

"""
    validate_buffers(u_new, u_old)

新旧バッファが安全に更新できる入力か検証する。

# 引数

- `u_new`: 更新結果を書き込む配列。旧配列と重ならない独立したバッファ。
- `u_old`: 更新前の値を読む配列。更新中は変更しない。

各軸3点以上・1始まりの同形状の浮動小数配列。非aliasとは同じ記憶領域を共有しないこと。
この条件と有限な旧値を検証する。不正入力は実装後の検証でArgumentError。

# 返り値

成功時はnothing。
"""
function validate_buffers(u_new, u_old)
    all(u -> u isa AbstractArray{<:AbstractFloat}, (u_new, u_old)) ||
        throw(ArgumentError("新旧バッファは浮動小数の配列です"))
    Base.require_one_based_indexing(u_new, u_old)
    axes(u_new) == axes(u_old) && ndims(u_old) > 0 && all(n -> n >= 3, size(u_old)) ||
        throw(ArgumentError("新旧バッファは同形状で各軸3点以上です"))
    (u_new === u_old || Base.mightalias(u_new, u_old)) &&
        throw(ArgumentError("新旧バッファが重なっています"))
    all(isfinite, u_old) || throw(ArgumentError("旧配列は有限値にしてください"))
    return nothing
end

"""
    fit_timestep(dt_max, t_final)

刻み上限を守り、最終時刻へ到達する刻みとステップ数を決める。

# 引数

- `dt_max`: 有限な正の刻み上限。
- `t_final`: 計算の最終時刻。

dt_max,t_finalは有限な正値。表現不能な刻みやステップ数も拒否する。

# 返り値

`(;steps, dt)` のNamedTuple。stepsは正整数、dtは有限正値でdt_max以下、steps回でt_finalへ到達する。
"""
function fit_timestep(dt_max, t_final)
    finite_real(dt_max, "dt_max")
    finite_real(t_final, "t_final")
    dt_max > 0 && t_final > 0 || throw(ArgumentError("dt_maxとt_finalは正です"))

    t, upper, _ = promote(t_final, dt_max, 1.0)
    steps = timestep_count(t / upper)
    dt = Float64(t / steps)
    if dt > dt_max
        # 切り上げやFloat64への丸めで上限を超えた場合は、上限側へ丸め直す。
        limit = Float64(dt_max)
        limit > dt_max && (limit = prevfloat(limit))
        limit > 0 || throw(ArgumentError("刻みがFloat64の表現範囲を外れています"))
        steps = timestep_count(BigFloat(t_final) / BigFloat(limit))
        dt = Float64(BigFloat(t_final) / steps)
    end
    isfinite(dt) && 0 < dt <= dt_max ||
        throw(ArgumentError("刻みがFloat64の表現範囲を外れています"))
    return (; steps, dt)
end

include("N01LinearAdvection.jl")
include("N02NonlinearAdvection.jl")
include("N03Diffusion.jl")
include("N04AdvectionDiffusion.jl")
include("N06Advection.jl")
include("N07Transport.jl")
include("N08N09Elliptic.jl")
end

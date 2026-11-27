module N04

import ..ThermofluidExercise as Common
import ..ThermofluidExercise: finite_real

validate_model(model) =
    model in (:linear, :nonlinear) ? nothing :
    throw(ArgumentError("SELECTED_MODELを:linearまたは:nonlinearに設定してください（model=$model）"))

function nonnegative_finite(v, name)
    finite_real(v, name)
    v >= 0 || throw(ArgumentError("$name は非負です"))
end

function positive_finite(v, name)
    finite_real(v, name)
    v > 0 || throw(ArgumentError("$name は正です"))
end

function validate_speed(model, speed)
    finite_real(speed, "speed")
    model == :linear && positive_finite(speed, "線形speed")
end

function validate_safety(safety)
    positive_finite(safety, "safety")
    safety <= 1 || throw(ArgumentError("safetyは0より大きく1以下です"))
end

function validate_timestep_inputs(a, dx, d, safety)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    nonnegative_finite(a, "max_speed")
    positive_finite(dx, "dx")
    nonnegative_finite(d, "diffusivity")
    validate_safety(safety)
    a > 0 ||
        d > 0 ||
        throw(ArgumentError("移流速度とdiffusivityがともに0では刻みを定義できません"))
end

function checked_timestep(dt)
    positive_finite(dt, "合成条件のdt（係数・dxの表現範囲を確認）")
    return dt
end

function maximum_speed(u, model, advection, speed)
    !advection && return 0.
    if model == :nonlinear
        minimum(u) >= 0 ||
            throw(ArgumentError("非線形移流は非負速度の範囲だけに対応します"))
        return maximum(u)
    end
    return speed
end

function stability_numbers(a, dt, dx, d)
    cfl = a * (dt / dx)
    fo = d * (dt / dx) / dx
    all(isfinite, (cfl, fo)) || throw(ArgumentError("CFL・Foが有限ではありません"))
    cfl + 2fo <= 1 + 32eps(Float64) ||
        throw(ArgumentError("合成安定条件CFL+2Fo<=1を超えています: CFL=$cfl, Fo=$fo"))
    return cfl, fo
end

function validate_step(a, b, dt, dx, d, model, advection, speed)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_model(model)
    validate_speed(model, speed)
    advection isa Bool || throw(ArgumentError("advectionはBoolです"))
    all(u->u isa AbstractVector{<:AbstractFloat}, (a, b)) ||
        throw(ArgumentError("新旧配列は浮動小数の一次元配列です"))
    Common.validate_buffers(a, b)
    positive_finite(dt, "dt")
    positive_finite(dx, "dx")
    nonnegative_finite(d, "diffusivity")
    stability_numbers(maximum_speed(b, model, advection, speed), dt, dx, d)
end

function advective_flux(u; model, speed = 1.0)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_model(model)
    finite_real(u, "u")
    validate_speed(model, speed)
    if model == :linear
        return Common.linear_flux(u, speed)
    else
        nonnegative_finite(u, "非線形移流のu")
        error("未実装 N04: advective_flux (nonlinear)")
    end
end

function stable_timestep(max_speed, dx, diffusivity; safety = 0.8)
    validate_timestep_inputs(max_speed, dx, diffusivity, safety)
    rate = max_speed / dx + 2 * (diffusivity / dx) / dx
    return checked_timestep(safety / rate)
end

function advection_diffusion_step!(
    u_new,
    u_old,
    dt,
    dx,
    diffusivity;
    model,
    advection = true,
    speed = 1.0,
)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_step(u_new, u_old, dt, dx, diffusivity, model, advection, speed)
    n = length(u_old)
    r = diffusivity * (dt / dx) / dx
    for i in eachindex(u_old)
        left = Common.periodic_left_index(i, n)
        right = Common.periodic_right_index(i, n)
        advective_change = advection ? (dt / dx) *
            (advective_flux(u_old[i]; model, speed) -
             advective_flux(u_old[left]; model, speed)) : 0.0
        u_new[i] = u_old[i] - advective_change +
                   r * (u_old[right] - 2 * u_old[i] + u_old[left])
    end
    return u_new
end

end

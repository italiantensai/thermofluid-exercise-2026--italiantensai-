module N03

import ..ThermofluidExercise as Common

positive_finite(v, name) =
    isfinite(v) && v > 0 ? nothing : throw(ArgumentError("$name は有限な正の値です"))

validate_boundary(boundary) =
    boundary in (:fixed, :insulated) ? nothing :
    throw(ArgumentError("boundaryは:fixedまたは:insulatedです"))

function validate_buffers(u_new, u_old, boundary)
    validate_boundary(boundary)
    all(u -> u isa AbstractVector{<:AbstractFloat}, (u_new, u_old)) ||
        throw(ArgumentError("新旧バッファは浮動小数の一次元配列です"))
    Common.validate_buffers(u_new, u_old)
end

function apply_boundary!(u_new, u_old, r::Real; boundary = :fixed)
    # 境界を書き込む前に、新旧配列と実効Fourier数を確認する。
    validate_buffers(u_new, u_old, boundary)
    positive_finite(r, "r")

    # 境界条件に応じて両端だけを更新する。
    if boundary == :fixed
        u_new[1] = 0.0
        u_new[end] = 0.0
    else
        u_new[1] = u_old[1] + 2 * r * (u_old[2] - u_old[1])
        u_new[end] = u_old[end] + 2 * r * (u_old[end - 1] - u_old[end])
    end

    return u_new
end

function diffusion_step!(
    u_new, u_old, dt::Real, dx::Real, diffusivity::Real; boundary = :fixed,
)
    # 配列の独立性と、刻み・格子幅・拡散係数を確認する。
    validate_buffers(u_new, u_old, boundary)
    for (value, name) in ((dt, "dt"), (dx, "dx"), (diffusivity, "diffusivity"))
        positive_finite(value, name)
    end

    # 更新に使う実効Fourier数を求め、表現可能な正値か確認する。
    r = diffusivity * dt / dx^2
    positive_finite(r, "実効Fourier数")

    # 内部点はすべて同じ旧配列から更新する。
    for i in 2:length(u_old)-1
        u_new[i] = u_old[i] + r * (u_old[i + 1] - 2 * u_old[i] + u_old[i - 1])
    end

    # 内部点の後で両端を更新し、次時刻の温度配列を完成させる。
    apply_boundary!(u_new, u_old, r; boundary)
    return u_new
end

end

module N02

import ..ThermofluidExercise as Common

function validate_boundary(boundary)
    boundary in (:fixed, :periodic) ||
        throw(ArgumentError("boundaryは:fixedまたは:periodicです"))
end

function validate_step_inputs(u_new, u_old, dt, dx, boundary)
    validate_boundary(boundary)
    all(u -> u isa AbstractVector{<:AbstractFloat}, (u_new, u_old)) ||
        throw(ArgumentError("新旧バッファは浮動小数の一次元配列です"))
    Common.validate_buffers(u_new, u_old)
    all(v -> isfinite(v) && v > 0, (dt, dx)) ||
        throw(ArgumentError("dtとdxは有限な正の値です"))
end

function nonlinear_upwind_step!(u_new, u_old, dt::Real, dx::Real; boundary = :fixed)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_step_inputs(u_new, u_old, dt, dx, boundary)
    copyto!(u_new, u_old)
    indices = boundary == :periodic ? (1:length(u_old)) : (2:(length(u_old) - 1))
    for i in indices
        left = boundary == :periodic ? Common.periodic_left_index(i, length(u_old)) : i - 1
        u_new[i] = u_old[i] - (dt / dx) *
                   (Common.burgers_flux(u_old[i]) - Common.burgers_flux(u_old[left]))
    end
    all(isfinite, u_new) || error("更新後に非有限値があります。更新式を確認してください")
    return u_new
end

function apply_boundary!(u; boundary = :fixed)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_boundary(boundary)
    length(u) >= 3 || throw(ArgumentError("境界配列には3点以上が必要です"))
    if boundary == :fixed
        u[1] = 1.0
        u[end] = u[end - 1]
    end
    return u
end

end

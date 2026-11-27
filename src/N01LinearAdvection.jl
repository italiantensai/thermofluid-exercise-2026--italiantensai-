module N01

function validate_step_inputs(u_new, u_old, c, dt, dx)
    u_new === u_old && throw(ArgumentError("新旧で別々のバッファを使ってください"))
    length(u_new) == length(u_old) >= 3 ||
        throw(ArgumentError("二つのバッファを同じ長さの3点以上にしてください"))
    all(isfinite, u_old) || throw(ArgumentError("u_oldの全要素を有限値にしてください"))
    all(isfinite, (c, dt, dx)) || throw(ArgumentError("c、dt、dxを有限値にしてください"))
    c > 0 || throw(ArgumentError("N01では正の移流速度だけを扱います"))
    dt > 0 || throw(ArgumentError("dtは正にしてください"))
    dx > 0 || throw(ArgumentError("dxは正にしてください"))
    return nothing
end

function validate_boundary_inputs(u, left_value)
    length(u) >= 2 || throw(ArgumentError("uには2点以上が必要です"))
    isfinite(left_value) || throw(ArgumentError("left_valueを有限値にしてください"))
    return nothing
end

function upwind_step!(u_new, u_old, c::Real, dt::Real, dx::Real)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_step_inputs(u_new, u_old, c, dt, dx)
    # courantは1ステップで進む格子幅の割合を表すCFL数です。
    courant = c * dt / dx
    # copyto!で古い値を複製し、内部を更新する間も両端点の値を保ちます。
    copyto!(u_new, u_old)
    for i in 2:(length(u_old) - 1)
        u_new[i] = u_old[i] - courant * (u_old[i] - u_old[i - 1])
    end
    return u_new
end

function centered_step!(u_new, u_old, c::Real, dt::Real, dx::Real)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_step_inputs(u_new, u_old, c, dt, dx)
    # courantは1ステップで進む格子幅の割合を表すCFL数です。
    courant = c * dt / dx
    # copyto!で古い値を複製し、内部を更新する間も両端点の値を保ちます。
    copyto!(u_new, u_old)
    for i in 2:(length(u_old) - 1)
        u_new[i] = u_old[i] - (courant / 2) * (u_old[i + 1] - u_old[i - 1])
    end
    return u_new
end

function apply_boundary!(u::AbstractVector{<:Real}; left_value::Real = 1.0)
    # 計算や書込みの前に、入力条件をまとめて確認する。
    validate_boundary_inputs(u, left_value)
    u[1] = left_value
    u[end] = u[end - 1]
    return u
end

end

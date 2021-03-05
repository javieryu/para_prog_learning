using LinearAlgebra

function proj_half(a, b, x)
    if a' * x <= b
        return x
    else
        return x - ((a' * x - b) / norm(a)^2) * a  
    end
end

function relu1(x)
    r = x
    r[1] = max(0, x[1])
    return r
end

function mine(a, b, x)
    n = size(a)[1]

    e1 = zeros(n)
    e1[1] = 1.0

    x0 = zeros(n)
    x0[1] = b / a[1]
    
    y = x - x0
    R = get_R(a, -e1)
    
    yrot = R * y
    return inv(R) * relu1(yrot) + x0
end

function get_R(x, y)
    n = size(x)[1]
    u = x / norm(x)
    v = y / norm(y)

    M = v * u' - u * v'
    M2 = M * M
    
    return I + v * u' - u * v' + (1 / (1 + x' * y)) * M2
end

function verify()
    n = 3
    a = randn(n)
    a = a / norm(a)
    b = 1.0 
    
    x = 3.0 * ones(n) + rand(n)
    
    r1 = proj_half(a, b, x)
    r2 = mine(a, b, x)
    
    println("a' * x <= b: ", a' * x <= b)
    println("Euc distance to stackexchange result: ", norm(r1 - r2))
end
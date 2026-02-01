## Single component


"""
    integrate(x::AbstractVector, y::AbstractVector, ::SimpsonFast)

Use Simpson's rule on an irregularly spaced grid x.
"""
function simpsons_rule(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || error("x and y vectors must be of the same length!")
    length(x) ≥ 2 || error("vectors must contain at least 3 elements")
    N = length(x)
    retval = zero(eltype(y))*x[1]
    yjp2 = y[1]
    for i in 0:floor(Int64, (N-1)/2)-1
        j = 2i+1
        jp1 = j+1
        jp2 = jp1+1

        dxj = x[jp1] - x[j]
        dxjp1 = x[jp2] - x[jp1]

        yj = yjp2
        yjp1 = y[jp1]
        yjp2 = y[jp2]

        term1 = (2 - dxjp1/dxj) * yj
        term2 = (dxjp1 + dxj)^2 / (dxjp1*dxj) * yjp1
        term3 = (2 - dxj/dxjp1) * yjp2

        retval += (dxj+dxjp1)*(term1+term2+term3)/6
    end
    if iseven(N)
        dxNm1 = x[N] - x[N-1]
        dxNm2 = x[N-1] - x[N-2]
        retval += (2dxNm1^2+3dxNm1*dxNm2) / (6(dxNm1+dxNm2)) * y[N]
        retval += (dxNm1^2+3dxNm1*dxNm2) / (6dxNm2) * y[N-1]
        retval -= (dxNm1^3) / (6dxNm2*(dxNm2+dxNm1)) * y[N-2]
    end
    return retval
end


"""
    compute_excess_energy(sol::OZSolution,  system::SimpleUnchargedSystem)

Computes the excess energy per particle Eₓ, such that E = (dims/2*kBT + Eₓ)*N.

uses the formula Eₓ = 1/2 ρ ∫dr g(r) u(r) for single component systems
and Eₓ = 1/2 ρ Σᵢⱼ xᵢxⱼ ∫dr gᵢⱼ(r) uᵢⱼ(r) for mixtures. Here x is the concentration fraction xᵢ=ρᵢ/sum(ρ).

"""
function compute_excess_energy(sol::OZSolution, system::System)
    dims = dims_of(system)
    Ns = number_of_species(system)
    r = sol.r
    ρ =  ρ_of(system)
    ρ0 = sum(ρ)
    x = get_concentration_fraction(system)
    gr = sol.gr
    if system isa SimpleUnchargedSystem
        u = evaluate_potential.(Ref(system.potential), r)
    elseif system isa SimpleChargedSystem
        basepot = base_of(system).potential
        u_base = evaluate_potential.(Ref(basepot), r)
        u_coul = evaluate_coulomb_potential(r, system) * kBT_of(system) # function returns in units of kBT
        u = u_base .+ u_coul
    else 
        error("This function doesn't support systems of type $(typeof(system))")
    end


    E = zero(eltype(eltype(gr)))

    rpow = dims-1
    sphere_surface = surface_N_sphere(dims)
    fraction_matrix = (x*x')

    for s1 = 1:Ns
        for s2 = 1:Ns
            integrand = gr[:, s1, s2] .* getindex.(u, s1, s2) .* r[:] .^ rpow 
            for i in eachindex(integrand) # if u is inf or gr is very small, set contribution to zero 
                if isnan(integrand[i]) || gr[i, s1, s2] < 10^-6
                    integrand[i] = zero(eltype(integrand))
                end
            end
            integral = simpsons_rule(r, integrand)
            E += fraction_matrix[s1,s2] * integral
        end
    end

    E *= sphere_surface*ρ0/2
    return E
end


function find_left_and_right_lim_y(potential, β, gr, r::AbstractArray, r0::Number)
    index = searchsortedfirst(r, r0) # first index >= disc
    rmin1 = r[index-1]
    rmin2 = r[index-2]
    ymin1 = exp.(β*evaluate_potential(potential, rmin1)).*gr[index-1, :, :]
    ymin2 = exp.(β*evaluate_potential(potential, rmin2)).*gr[index-2, :, :]
    yleft = ymin1 + (r0 - rmin1) * (ymin2 - ymin1) / (rmin2 - rmin1)
    
    r1 = r[index+1]
    r2 = r[index+2]
    y1 = exp.(β*evaluate_potential(potential, r1)).*gr[index+1, :, :]
    y2 = exp.(β*evaluate_potential(potential, r2)).*gr[index+2, :, :]
    yright = y1 + (r0 - r1) * (y2 - y1) / (r2 - r1)
    return yleft, yright
end

function find_de_mul_y0(discontinuity, β, potential, gr, r)
    eleft = exp.(-β*evaluate_potential(potential, prevfloat(discontinuity)))
    eright = exp.(-β*evaluate_potential(potential, nextfloat(discontinuity)))
    de = eright - eleft
    yleft, yright = find_left_and_right_lim_y(potential, β, gr, r, discontinuity)

    if all(isfinite.(yleft)) && all(isfinite.(yright))
        ydisc = (yleft + yright)/2
        if abs(yleft - yright) > 0.1
            error("This is weird, the cavity distribution function looks discontinuous")
        end
        return de.*ydisc
    elseif all(isfinite.(yleft))
        ydisc = yleft
        return de.*ydisc
    elseif all(isfinite.(yright))
        ydisc = yright
        return de.*ydisc
    else
        @show yleft, yright
        error("This cannot happen. File an issue")
    end
end


# surface of sphere embedded in n dimensions d=3->4pi
function surface_N_sphere(n)
    return 2π^(n/2)/gamma(n/2)
end


"""
    compute_virial_pressure(sol::OZSolution,  system::SimpleUnchargedSystem)

Computes the pressure via the virial route

uses the formula p = kBTρ - 1/(2d) ρ^2 ∫dr r g(r) u'(r) for single component systems
and p =  kBT Σᵢρᵢ - 1/(2d) Σᵢⱼ ρᵢρⱼ ∫dr r gᵢⱼ(r) u'ᵢⱼ(r) for mixtures.

It handles discontinuities in the interaction potential analytically if `discontinuities(potential)` is defined.
For additional speed/accuracy define a method of `evaluate_potential_derivative(potential, r::Number)` that analytically computes du/dr. 
By default this is done using finite differences.
"""
function compute_virial_pressure(sol::OZSolution, system:: SimpleUnchargedSystem) 
    dims = dims_of(system)
    Ns = number_of_species(system)
    r = sol.r
    ρ = system.ρ
    ρ0 = sum(ρ)
    x = get_concentration_fraction(system)
    kBT = system.kBT
    β = 1/kBT
    gr = sol.gr
    potential = system.potential
    dudr = evaluate_potential_derivative(potential, r)
    rpow = dims
    p1 = zero(eltype(gr))
    fraction_matrix = (x*x')
    for s1 = 1:Ns
        for s2 = 1:Ns
            integrand = gr[:, s1, s2] .* getindex.(dudr, s1, s2) .* r[:] .^ rpow 
            for i in eachindex(integrand) # if u is inf or gr is very small, set contribution to zero 
                if isnan(integrand[i]) || gr[i, s1, s2] < 10^-6
                    integrand[i] = zero(eltype(integrand))
                end
            end
            integral = simpsons_rule(r, integrand)
            p1 += fraction_matrix[s1,s2]*integral
        end
    end
    sphere_surface = surface_N_sphere(dims)
    p = kBT*ρ0 - sphere_surface/(2*dims) * ρ0^2 * p1

    ## now add the terms for the discontinuities

    discs = unique(discontinuities(system.potential))
    for discontinuity in discs
        dey0 = find_de_mul_y0(discontinuity, β, potential, gr, r)
        dp = (sphere_surface*ρ0^2)/((2*dims)*β)*discontinuity^rpow*sum((x*x') .* dey0)
        p += dp
    end

    return p
end


"""
    compute_virial_pressure(sol::OZSolution,  system::SimpleUnchargedSystem)

Computes the pressure via the virial route

uses the formula p = kBTρ - 1/(2d) ρ^2 ∫dr r g(r) u'(r) for single component systems
and p =  kBT Σᵢρᵢ - 1/(2d) Σᵢⱼ ρᵢρⱼ ∫dr r gᵢⱼ(r) u'ᵢⱼ(r) for mixtures.

It handles discontinuities in the interaction potential analytically if `discontinuities(potential)` is defined.
For additional speed/accuracy define a method of `evaluate_potential_derivative(potential, r::Number)` that analytically computes du/dr. 
By default this is done using finite differences.
"""
function compute_virial_pressure_charged(sol::OZSolution, system:: SimpleChargedSystem) 

    dims = dims_of(system)
    Ns = number_of_species(system)
    r = sol.r
    ρ =  ρ_of(system)
    ρ0 = sum(ρ)
    x = get_concentration_fraction(system)
    kBT = kBT_of(system)
    β = 1/kBT
    gr = sol.gr
    basepot = base_of(system).potential
    du_base = evaluate_potential_derivative.(Ref(basepot), r)
    du_coul = evaluate_coulomb_potential_derivative(r, system) * kBT_of(system) # function returns in units of kBT
    dudr = du_base .+ du_coul
    rpow = dims
    p1 = zero(eltype(gr))
    fraction_matrix = (x*x')
    for s1 = 1:Ns
        for s2 = 1:Ns
            integrand = gr[:, s1, s2] .* getindex.(dudr, s1, s2) .* r[:] .^ rpow 
            for i in eachindex(integrand) # if u is inf or gr is very small, set contribution to zero 
                if isnan(integrand[i]) || gr[i, s1, s2] < 10^-6
                    integrand[i] = zero(eltype(integrand))
                end
            end
            integral = simpsons_rule(r, integrand)
            p1 += fraction_matrix[s1,s2]*integral
        end
    end
    sphere_surface = surface_N_sphere(dims)
    p = kBT*ρ0 - sphere_surface/(2*dims) * ρ0^2 * p1

    ## now add the terms for the discontinuities

    discs = unique(discontinuities(base_of(system).potential))
    for discontinuity in discs
        dey0 = find_de_mul_y0(discontinuity, β, base_of(system).potential, gr, r)
        dp = (sphere_surface*ρ0^2)/((2*dims)*β)*discontinuity^rpow*sum((x*x') .* dey0)
        p += dp
    end

    return p
end

function get_concentration_fraction(system::System)
    ρ = ρ_of(system)
    if ρ isa AbstractArray
        return ρ.diag / sum(ρ.diag)
    elseif ρ isa Number
        return one(ρ)
    end
    error("Unreachable code: file an issue!")
end

_eachslice(a;dims=1) = eachslice(a,dims=dims)
_eachslice(a::Vector;dims=1) = a


"""
    compute_compressibility_charged(sol::OZSolution, system::SimpleFluid)

Computes the isothermal compressibility χ of the system

uses the formula 1/ρkBTχ = 1 - ρ ĉ(k=0) for single component systems and
1/ρkBTχ = 1 - ρ Σᵢⱼ ĉᵢⱼ(k=0) for mixtures. 
Eq. (3.6.16) in Hansen and McDonald
"""
function compute_compressibility_charged(sol::OZSolution, system::SimpleChargedSystem) 
    ρ = ρ_of(system)#system.ρ
    kBT = kBT_of(system)#system.kBT
    x = get_concentration_fraction(system)
    ρ0 = sum(ρ)
    T = typeof(ρ0)
    ĉ0 = get_ĉ0(sol, system)
    invρkBTχ = one(T) - ρ0 * sum((x*x') .* ĉ0)
    χ = (one(T)/invρkBTχ)/(kBT*ρ0)
    return χ
end

"""
    compute_compressibility(sol::OZSolution, system::SimpleFluid)

Computes the isothermal compressibility χ of the system

uses the formula 1/ρkBTχ = 1 - ρ ĉ(k=0) for single component systems and
1/ρkBTχ = 1 - ρ Σᵢⱼ ĉᵢⱼ(k=0) for mixtures. 
Eq. (3.6.16) in Hansen and McDonald
"""
function compute_compressibility(sol::OZSolution, system::SimpleUnchargedSystem) 
    ρ = system.ρ
    kBT = system.kBT
    x = get_concentration_fraction(system)
    ρ0 = sum(ρ)
    T = typeof(ρ0)
    ĉ0 = get_ĉ0(sol, system)
    invρkBTχ = one(T) - ρ0 * sum((x*x') .* ĉ0)
    χ = (one(T)/invρkBTχ)/(kBT*ρ0)
    return χ
end

function get_ĉ0(sol::OZSolution, system::SimpleFluid) 
    dims = dims_of(system)
    spl = Spline1D(sol.r, sol.cr[:, 1, 1].*sol.r.^(dims-1))
    ĉ0 = surface_N_sphere(dims)*integrate(spl, zero(eltype(sol.r)), maximum(sol.r))
    return ĉ0
end

function get_ĉ0(sol::OZSolution, s::SimpleMixture)
    dims = dims_of(s)
    species = number_of_species(s)
    ĉ0 = zeros(eltype(eltype(sol.ck)), species, species)
    for i = 1:species
        for j = 1:species
            spl = Spline1D(sol.r, sol.cr[:, i, j].*sol.r.^(dims-1))
            ĉ0[i,j] = surface_N_sphere(dims)*integrate(spl, zero(eltype(sol.r)), maximum(sol.r))
        end
    end
    return ĉ0
end

function get_ĉ0(sol::OZSolution, s::SimpleChargedMixture)
    dims = dims_of(s)
    species = number_of_species(s)
    ĉ0 = zeros(eltype(eltype(sol.ck)), species, species)
    for i = 1:species
        for j = 1:species
            spl = Spline1D(sol.r, sol.cr[:, i, j].*sol.r.^(dims-1))
            ĉ0[i,j] = surface_N_sphere(dims)*integrate(spl, zero(eltype(sol.r)), maximum(sol.r))
        end
    end
    return ĉ0
end

function compute_activity_coefficient_c(sol::OZSolution, system::System)
    r = sol.r
    cr = sol.cr
    
    dims = dims_of(system)
    Ns = number_of_species(system)
    
    ρ = ρ_of(system)
    ρ_total = sum(ρ)
    x = get_concentration_fraction(system)
    
    sphere_surface = surface_N_sphere(dims)
    rpow = dims - 1
    
    # Compute integrals of cᵢⱼ(r)
    c_integrals = zeros(eltype(r), Ns, Ns)
    
    for i in 1:Ns
        for j in 1:Ns
            integrand = cr[:, i, j] .* r[:] .^ rpow
            
            for k in eachindex(integrand)
                if isnan(integrand[k]) || !isfinite(integrand[k])
                    integrand[k] = zero(eltype(integrand))
                end
            end
            
            c_integrals[i, j] = sphere_surface * simpsons_rule(r, integrand)
        end
    end
    
    # Compute individual activity coefficients first
    # ln(γᵢ) = -ρ_total Σⱼ xⱼ ∫cᵢⱼ r^(d-1) dr
    ln_γ = zeros(eltype(r), Ns)
    
    for i in 1:Ns
        for j in 1:Ns
            ln_γ[i] -= ρ_total * x[j] * c_integrals[i, j]
        end
    end
    
    # Compute mean ionic activity coefficient
    # For electrolytes: ln(γ±) = (Σᵢ νᵢ ln(γᵢ)) / (Σᵢ νᵢ)
    #z = system.z
    #ν = abs.(z) ./ gcd(abs.(Int.(z))...)
    #ln_γ_mean = sum(ν .* ln_γ) / sum(ν)

    z = system.z
    ν₊ = abs(z[2])  # ν₊ = |z₋|
    ν₋ = abs(z[1])  # ν₋ = |z₊|
    ν_total = ν₊ + ν₋
    
    ln_γ_mean = (ν₊ * ln_γ[1] + ν₋ * ln_γ[2]) / ν_total
    
    return ln_γ_mean
end

# --- helper: Simpson weights (fallback if OrnsteinZernike.simpsons_weights not available) ---
function simpson_weights(r::AbstractVector)
    n = length(r)
    if n < 3
        return ones(eltype(r), n)
    end
    h = diff(r)
    # non-uniform spacing Simpson-like weights (composite Simpson for uniform spacing is trivial)
    # We'll produce trapezoid-like weights when spacing not uniform, but prefer uniform Simpson if uniform.
    if maximum(abs.(h .- h[1])) < 1e-12
        # uniform spacing
        Δ = h[1]
        w = zeros(eltype(r), n)
        w[1] = 1/3
        w[end] = 1/3
        for i in 2:n-1
            w[i] = (i % 2 == 0) ? 4/3 : 2/3
        end
        return w .* Δ
    else
        # fallback to trapezoidal weights for uneven spacing (safe)
        w = zeros(eltype(r), n)
        w[1] = h[1] / 2
        for i in 2:n-1
            w[i] = (h[i-1] + h[i]) / 2
        end
        w[end] = h[end] / 2
        return w
    end
end

# --- Optimized compute_activity_coefficient ---
function compute_activity_coefficient_fast(sol::OZSolution, system::System, closure::Closure; threaded::Bool=true)
    # unpack frequently-used arrays
    r = sol.r                       # vector, length Nr
    cr = sol.cr                     # Nr × Ns × Ns
    gr = sol.gr                     # Nr × Ns × Ns
    gamma_r = sol.gamma_r           # Nr × Ns × Ns

    dims = OrnsteinZernike.dims_of(system)
    Ns = OrnsteinZernike.number_of_species(system)
    ρ = OrnsteinZernike.ρ_of(system)
    ρ_total = sum(ρ)
    x = OrnsteinZernike.get_concentration_fraction(system)

    sphere_surface = OrnsteinZernike.surface_N_sphere(dims)
    rpow = dims - 1

    Nr = length(r)

    # precompute r weight r^(d-1) once
    rweight = similar(r)
    @inbounds @fastmath for k in 1:Nr
        rweight[k] = r[k]^rpow
    end

    # precompute Simpson / integration weights once
    # Prefer an OrnsteinZernike-provided weights function if present
    w = try
        OrnsteinZernike.simpsons_weights(r)
    catch
        simpson_weights(r)
    end

    # 1) Compute potential u(r) and mayer f once (hoisted)
    β = 1.0 / OrnsteinZernike.kBT_of(system)

    # Evaluate potential once (handle charged/uncharged)
    u = similar(r)
    if system isa SimpleUnchargedSystem
        u .= OrnsteinZernike.evaluate_potential.(Ref(system.potential), r)
    elseif system isa SimpleChargedSystem || system isa SimpleChargedMixture
        # try to use base potential inside system and add coulomb term
        basepot = OrnsteinZernike.base_of(system).potential
        u_base = OrnsteinZernike.evaluate_potential.(Ref(basepot), r)
        u_coul = OrnsteinZernike.evaluate_coulomb_potential(r, system) .* OrnsteinZernike.kBT_of(system)
        #@inbounds @fastmath for k in 1:Nr
        #    u[k] = u_base[k] + u_coul[k]
        #end
        u = u_base .+ u_coul
    else
        error("System type not supported in compute_activity_coefficient_fast")
    end

    # βu and Mayer f
    #βu = similar(r)
    #@inbounds @fastmath for k in 1:Nr
    #    βu[k] = β * u[k]
    #end
    βu = β .* u
    mayer_f = OrnsteinZernike.find_mayer_f_function.(βu)   # vectorized dispatch; if expensive, cache externally

    # 2) Bridge function (hoisted)
    #B_r = OrnsteinZernike.bridge_function(closure, r, mayer_f, gamma_r)

    # 3) prepare outputs and temporary storage (preallocated)
    ln_γ_species = zeros(eltype(r), Ns)
    integrand = similar(r)   # reuse for every i,j pair

    # inner accumulation loop: thread over species i (safe because each i writes a unique element in ln_γ_species)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:Ns
            integral_sum = zero(eltype(r))

            # local alias to avoid globals inside threaded loop
            @inbounds @fastmath for j in 1:Ns
                h_view = @view gr[:, i, j]
                c_view = @view cr[:, i, j]
                # gamma view if you want to include gamma in integrand variants:
                # γ_view = @view gamma_r[:, i, j]
                #b_view = @view B_r[:, i, j]

                # compute integrand into preallocated buffer
                @inbounds @fastmath for k in 1:Nr
                    # compute h(r) = g(r) - 1 on the fly (avoid alloc)
                    h = h_view[k] - 1.0
                    c = c_view[k]
                    # chosen integrand: 0.5*h^2 - c - 0.5*c*h  (matches your current line)
                    val = 0.5*h*h - c - 0.5*c*h
                    # include b_view[k] terms if you want: val += b_view[k] + 2.0/3.0*h*b_view[k]
                    # safe guard for NaN/Inf
                    integrand[k] = isfinite(val) ? (val * rweight[k]) : zero(eltype(val))
                end

                # weighted dot product with Simpson weights
                s = zero(eltype(r))
                @inbounds for k in 1:Nr
                    s += integrand[k] * w[k]
                end

                integral_val = sphere_surface * s
                integral_sum += (ρ_total * x[j]) * integral_val
            end

            ln_γ_species[i] = integral_sum
        end
    else
        # single-threaded fallback
        for i in 1:Ns
            integral_sum = zero(eltype(r))
            @inbounds @fastmath for j in 1:Ns
                h_view = @view gr[:, i, j]
                c_view = @view cr[:, i, j]
                #b_view = @view B_r[:, i, j]

                @inbounds @fastmath for k in 1:Nr
                    h = h_view[k] - 1.0
                    c = c_view[k]
                    val = 0.5*h*h - c - 0.5*c*h
                    integrand[k] = isfinite(val) ? (val * rweight[k]) : zero(eltype(val))
                end

                s = zero(eltype(r))
                @inbounds for k in 1:Nr
                    s += integrand[k] * w[k]
                end

                integral_val = sphere_surface * s
                integral_sum += (ρ_total * x[j]) * integral_val
            end
            ln_γ_species[i] = integral_sum
        end
    end

    # 4) mean ionic activity coefficient / selection logic (unchanged)
    z = OrnsteinZernike.zvec_of(system)
    if length(z) >= 2 && any(!iszero, z)
        ln_γ_mean = sum(ρ .* ln_γ_species) / ρ_total
        return ln_γ_mean
    else
        return ln_γ_species[1]
    end
end

function compute_activity_coefficient(sol::OZSolution, system::System, closure::Closure)
    r = sol.r
    cr = sol.cr
    gr = sol.gr 
    gamma_r = sol.gamma_r 
    
    dims = OrnsteinZernike.dims_of(system)
    Ns = OrnsteinZernike.number_of_species(system)
    
    ρ = OrnsteinZernike.ρ_of(system)
    ρ_total = sum(ρ)
    x = OrnsteinZernike.get_concentration_fraction(system)
    
    sphere_surface = OrnsteinZernike.surface_N_sphere(dims)
    rpow = dims - 1
    
    # 1. Calculate Mayer f-function
    if system isa SimpleUnchargedSystem
        u = evaluate_potential.(Ref(system.potential), r)
    elseif system isa SimpleChargedSystem
        basepot = OrnsteinZernike.base_of(system).potential
        u_base = OrnsteinZernike.evaluate_potential.(Ref(basepot), r)
        u_coul = OrnsteinZernike.evaluate_coulomb_potential(r, system) * OrnsteinZernike.kBT_of(system) 
        u = u_base .+ u_coul
    else 
        error("System type not supported")
    end
    
    β = 1.0 / OrnsteinZernike.kBT_of(system)
    βu = u .* β
    mayer_f = OrnsteinZernike.find_mayer_f_function.(βu)

    # 2. Compute Bridge Function B(r)
    #B_r = OrnsteinZernike.bridge_function(closure, r, mayer_f, gamma_r)

    # Storage for results
    ln_γ_species = zeros(eltype(r), Ns)
    
    # 3. Integration Loop
    for i in 1:Ns 
        integral_sum = 0.0
        
        for j in 1:Ns 
            h_ij = gr[:, i, j] .- 1.0
            c_ij = cr[:, i, j]
            γ_ij = gamma_r[:, i, j]
            g_ij = gr[:, i, j]
            #b_ij = B_r[:, i, j]
            
            integrand = @. (0.5 .* (h_ij .^ 2) .- c_ij .- 0.5 * c_ij .* h_ij) .* r .^ rpow

            #integrand = @. (g_ij .* log(abs(g_ij)) .- g_ij .+ 1 .+ kBT .* g_ij .* u_ij) .* r .^ rpow
            
            for k in eachindex(integrand)
                if isnan(integrand[k]) || !isfinite(integrand[k])
                    integrand[k] = zero(eltype(integrand))
                end
            end
            integral_val = sphere_surface * OrnsteinZernike.simpsons_rule(r, integrand)
            integral_sum += (ρ_total * x[j]) * integral_val
        end
        ln_γ_species[i] = integral_sum
    end
     
    #z = OrnsteinZernike.zvec_of(system)
    #if length(z) >= 2 && any(!iszero, z)
    #    ln_γ_mean = sum(ρ .* ln_γ_species) / ρ_total
    #    return ln_γ_mean
    #else
    #    return ln_γ_species[1]
    #end
    z = OrnsteinZernike.zvec_of(system)
    
    if any(!iszero, z)   # electrolyte
        z_plus  = findfirst(z .> 0)
        z_minus = findfirst(z .< 0)

        ν_plus  = abs(z[z_minus])
        ν_minus = abs(z[z_plus])
        ν_tot   = ν_plus + ν_minus

        ln_γ_mean = (ν_plus * ln_γ_species[z_plus] +
                    ν_minus * ln_γ_species[z_minus]) / ν_tot
        return ln_γ_mean

    else
        # one-component case
        return ln_γ_species[1]
    end
end

"""
    charging_route_muex(u, r, density; T, nλ=20, hnc_solver)

Compute the excess chemical potential μ^ex using the charging route:

    μ^ex = ∫₀¹ dλ ∫ 4π r² ρ g_λ(r) u(r) dr

Arguments
---------
- `u`          : potential u(r) as a vector (same length as r)
- `r`          : radial grid (vector)
- `density`    : number density ρ of the bath species
- `T`          : temperature (K)
- `nλ`         : number of λ integration steps (default 20)
- `hnc_solver` : a function (uλ, r, density, T) → gλ

Returns
-------
- μ_ex : excess chemical potential (in k_B*T units)
"""
function charging_route_muex(system, closure, imethod; T=298.15, nλ=50)

    kBT = OrnsteinZernike.kBT_of(system)#1.380649e-23 * T   # Joules
    density = sum(OrnsteinZernike.ρ_of(system))
    ρ_vec = diag(OrnsteinZernike.ρ_of(system))
    Ns = OrnsteinZernike.number_of_species(system)
    dims = OrnsteinZernike.dims_of(system)
    #dr = r[2] - r[1]
    λ_values = range(1e-3, 1; length=nλ)
    #λ_values[1] = λ_values[1] + 1e-3
    
    integrand_λ = zeros(nλ)

    for (iλ, λv) in enumerate(λ_values)
        #uλ = λ .* u

        #pref = kB * B / σ / kBT   # matches earlier kB*B/σ*Sones/kBT
        #soft = OrnsteinZernike.InversePowerLawMixture(λ .* pref .* ϵ_vec, σ_vec ./ σ, n)
        #pref_gauss = 1/pref
        #g1 = OrnsteinZernike.Gaussian(λ .* pref_gauss .* [ϵ_gauss, ϵ_gauss], [σ_gauss, σ_gauss], [μ_gauss, μ_gauss])
        #g2 = OrnsteinZernike.Gaussian(λ .* pref_gauss .* [ϵ_gauss2, ϵ_gauss2], [σ_gauss2, σ_gauss2], [μ_gauss2, μ_gauss2])
        #coul = OrnsteinZernike.CustomCoulomb(Z, λ .* bjerrum_length / σ)
        #pot = OrnsteinZernike.CompositePotential(soft, g1, g2)#, coul)
        #pot = OrnsteinZernike.CompositePotential(soft, g1, g2, coul)
        pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss, 
                           ϵ_gauss2, σ_gauss2, μ_gauss2, Z, false, λ=λv)

        system = SimpleMixture(dims, ρ_vec, 1, pot)
        system = SimpleChargedMixture(system, Z, λv * bjerrum_length / σ)

        # call your HNC/OZ solver to get g(r; λ)
        #gλ = hnc_solver(uλ, r, density, T)
        sol = solve(system, closure, imethod)#,
                #coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))

        # Extract RDFs
        r = sol.r 
        gλ = sol.gr
        dr = r[2] - r[1]

        #basepot = OrnsteinZernike.base_of(system).potential
        #u_base = OrnsteinZernike.evaluate_potential.(Ref(basepot), r)
        #u_coul = OrnsteinZernike.evaluate_coulomb_potential(r, system) * OrnsteinZernike.kBT_of(system) # function returns in units of kBT
        #u = u_base .+ u_coul

        u = OrnsteinZernike.evaluate_potential.(Ref(OrnsteinZernike.base_of(system).potential), r)
        u .+= OrnsteinZernike.evaluate_coulomb_potential(r, system) .* kBT
        
        # radial integral: ∫ 4π r² ρ g_λ(r) u(r) dr
        #integrand_λ[i] = 4π * density * sum(r.^2 .* gλ .* u) * dr
        # ---- radial integral over all pairs (i,j) ----
        sum_pairs = 0.0
        for i in 1:Ns
            for j in 1:Ns
                sum_pairs += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* getindex.(u, i, j)) * dr
            end
        end
        integrand_λ[iλ] = 4π * sum_pairs
        #println(i," ",λv)
    end

    # integrate over λ using Simpson rule
    μ_ex_J = OrnsteinZernike.simpsons_rule(λ_values, integrand_λ)   # in Joules

    return lnγ_mean = μ_ex_J / (2 * kBT) #μ_ex_J / kBT      # return in k_B*T units
end



function charging_route_muex_2(system, closure, imethod; T=298.15, nλ=50)
    kBT = OrnsteinZernike.kBT_of(system)
    ρ_vec = diag(OrnsteinZernike.ρ_of(system))
    Ns = OrnsteinZernike.number_of_species(system)
    dims = OrnsteinZernike.dims_of(system)
    
    #nλ = 1000
    λ_values = range(1e-3, 1; length=nλ)
    #λ_values = [0.001 .+ (0.999) .* ((i-1)/(nλ-1))^2 for i in 1:nλ]
    integrand_λ = zeros(nλ)

    gamma_r = nothing

    imethod = NgIteration(M = M,
                          dr = dr,
                          tolerance = 1e-6,
                          verbose = false,
                          max_iterations = 10^6,
                          N_stages = 8)

    μ_ex_J = 0.0
    for (iλ, λv) in enumerate(λ_values)
        pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss, 
                           ϵ_gauss2, σ_gauss2, μ_gauss2, Z, false, λ=λv)

        system = SimpleMixture(dims, ρ_vec, 1, pot)
        system = SimpleChargedMixture(system, Z, λv * bjerrum_length / σ)

        if iλ == 1
            sol = solve(system, closure, imethod; coulombsplitting=EwaldSplitting(3.0))
        else
            sol = solve(system, closure, imethod, gamma_0=gamma_r; coulombsplitting=EwaldSplitting(3.0))
        end

        r = sol.r 
        gλ = sol.gr
        gamma_r = convert_3darr_to_vec_of_smat(sol.gamma_r)
        dr = r[2] - r[1]

        u = OrnsteinZernike.evaluate_potential.(Ref(OrnsteinZernike.base_of(system).potential), r)
        u .+= OrnsteinZernike.evaluate_coulomb_potential(r, system) .* kBT

        # Divide by λv to get unscaled u(r)
        u ./= λv
        
        # Sum over unique pairs to avoid double counting
        sum_pairs = 0.0
        for i in 1:Ns
            for j in i:Ns  # j >= i to avoid double counting
                factor = (i == j) ? 1.0 : 2.0  # off-diagonal terms counted twice
                sum_pairs += factor * ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* getindex.(u, i, j)) * dr
            end
        end
        integrand_λ[iλ] = 4π * sum_pairs

        #μ_ex_J_old = μ_ex_J
        #if iλ>1
        #    dλ = λ_values[iλ] - λ_values[iλ-1]
        #    μ_ex_J += 0.5 * dλ * (integrand_λ[iλ] + integrand_λ[iλ-1])
        #end
        #err_μ_ex = abs(μ_ex_J - μ_ex_J_old)/abs(μ_ex_J_old)
        #println(iλ," ",err_μ_ex," ",μ_ex_J)
    end

    #μ_ex_J = OrnsteinZernike.simpsons_rule(λ_values, integrand_λ)
    #μ_ex_J, err = quadgk(λ -> compute_integrand(λ), 0.001, 1.0, rtol=1e-6)
    μ_ex_J = 0.0
    for i in 1:(nλ-1)
        dλ = λ_values[i+1] - λ_values[i]
        μ_ex_J += 0.5 * dλ * (integrand_λ[i] + integrand_λ[i+1])
    end
    μ_ex_J
    
    return lnγ_mean = μ_ex_J / (2 * kBT)
end

function charging_route_muex_3(system, closure, imethod; T=298.15, rtol=1e-6)
    kBT = OrnsteinZernike.kBT_of(system)
    ρ_vec = diag(OrnsteinZernike.ρ_of(system))
    Ns = OrnsteinZernike.number_of_species(system)
    dims = OrnsteinZernike.dims_of(system)
    
    gamma_prev = Ref{Any}(nothing)

    imethod = NgIteration(M = M,
                          dr = dr,
                          tolerance = 1e-6,
                          verbose = false,
                          max_iterations = 10^6,
                          N_stages = 8)
    
    # Define the integrand as a function of λ
    function compute_integrand(λv)
        pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss, 
                           ϵ_gauss2, σ_gauss2, μ_gauss2, Z, false, λ=λv)

        sys = SimpleMixture(dims, ρ_vec, 1, pot)
        sys = SimpleChargedMixture(sys, Z, λv * bjerrum_length / σ)

        if isnothing(gamma_prev[])
            #println("wow ", λv)
            sol = solve(sys, closure, imethod; coulombsplitting=EwaldSplitting(3.0))
        else
            #println("uau ",λv)
            try 
                sol = solve(sys, closure, imethod; gamma_0=gamma_prev[], coulombsplitting=EwaldSplitting(3.0))
            catch
                sol = solve(sys, closure, imethod; coulombsplitting=EwaldSplitting(3.0))
            end
        end

        r = sol.r 
        gλ = sol.gr
        #gamma_prev = convert_3darr_to_vec_of_smat(sol.gamma_r)
        gamma_prev[] = convert_3darr_to_vec_of_smat(sol.gamma_r)
        dr = r[2] - r[1]

        u = OrnsteinZernike.evaluate_potential.(Ref(OrnsteinZernike.base_of(sys).potential), r)
        u .+= OrnsteinZernike.evaluate_coulomb_potential(r, sys) .* kBT
        u ./= λv  # Get unscaled potential
        
        # Sum over unique pairs
        sum_pairs = 0.0
        for i in 1:Ns
            for j in i:Ns
                factor = (i == j) ? 1.0 : 2.0
                sum_pairs += factor * ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* getindex.(u, i, j)) * dr
            end
        end
        
        return 4π * sum_pairs
    end
    
    # Adaptive integration from λ=0.001 to λ=1
    rtol = 1e-2
    μ_ex_J, err = quadgk(compute_integrand, 0.001, 0.999; rtol=rtol)
    
    println("Integration error estimate: $err")
    
    return μ_ex_J / (2 * kBT)
end

function convert_3darr_to_vec_of_smat(A::Array{T,3}) where {T<:Real}
    Nr, Ns1, Ns2 = size(A)
    out = Vector{SMatrix{Ns1,Ns2,T,Ns1*Ns2}}(undef, Nr)
    for i in 1:Nr
        out[i] = SMatrix{Ns1,Ns2}(A[i, :, :])
    end
    return out
end

function activity_coefficient_routes(c; T=298.15, nλ=50)
    #F-route
    ρ0 = c * 6.02214076e23 * 1e3  
    ρ = [ρ0, ρ0] # equal density of cations and anions
    # Define short-range repulsive potential
    hs = HardSpheres(σ_vec .* 1e10)
    pref = 0.01#B / σ / T   # matches earlier kB*B/σ*Sones/kBT
    soft = InversePowerLawMixture(pref .* ϵ_vec, σ_vec ./ σ, n)
    pref_gauss = 0#1/pref
    g1 = Gaussian(pref_gauss .* [ϵ_gauss, ϵ_gauss], [σ_gauss, σ_gauss], [μ_gauss, μ_gauss])
    g2 = Gaussian(pref_gauss .* [ϵ_gauss2, ϵ_gauss2], [σ_gauss2, σ_gauss2], [μ_gauss2, μ_gauss2])
    g3 = Gaussian(pref_gauss .* [ϵ_gauss3, ϵ_gauss3], [σ_gauss3, σ_gauss3], [μ_gauss3, μ_gauss3])
    g4 = Gaussian(pref_gauss .* [ϵ_gauss4, ϵ_gauss4], [σ_gauss4, σ_gauss4], [μ_gauss4, μ_gauss4])
    #coul = CustomCoulomb(Z, bjerrum_length / σ)
    # Composite Potential
    pot = CompositePotential(soft, g1, g2, g3, g4)#, coul)
    #pot0 = CompositePotential(soft, g1, g2, g3, g4, coul)
    # Build system and add Coulomb interactions
    system = SimpleMixture(dims, ρ*σ^3, 1, pot)
    system = SimpleChargedMixture(system, Z, bjerrum_length / σ)
    # Choose closure and numerical method
    closure = HypernettedChain()#HypernettedChain()
    ngmethod  = NgIteration(M = M,
                          dr = dr,
                          tolerance = 1e-8,
                          verbose = false,
                          max_iterations = 10^6,
                          N_stages = 8)
    # Solve OZ equation with HNC closure
    sol = solve(system, closure, ngmethod)#,
                #coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
    r = sol.r 
    g = sol.gr
    Ex = OrnsteinZernike.compute_activity_coefficient(sol, system, closure)
    println("F-route: ",Ex)

    system = SimpleMixture(dims, ρ*σ^3, 1, pot0)
    sol = solve(system, closure, ngmethod)
    r = sol.r 
    g = sol.gr
    Ex = OrnsteinZernike.compute_activity_coefficient(sol, system, closure)
    println("F-route pot0: ",Ex)

    #Charging route
    for nλ in [10,15,20,25,50,100,200,500,1000,2000,5000,10000]
        #nλ = 200
        #println(nλ)
        kBT = OrnsteinZernike.kBT_of(system)
        ρ_vec = diag(OrnsteinZernike.ρ_of(system))
        Ns = OrnsteinZernike.number_of_species(system)
        dims = OrnsteinZernike.dims_of(system)
        #λ_values = range(1e-3, 1; length=nλ)
        λ_values = [0.0001 .+ (0.999) .* ((i-1)/(nλ-1))^2 for i in 1:nλ]
        integrand_λ = zeros(nλ)
        gamma_r = nothing
        imethod = NgIteration(M = M,
                            dr = dr,
                            tolerance = 1e-6,
                            verbose = false,
                            max_iterations = 10^6,
                            N_stages = 4)
        μ_ex_J = 0.0
        # Preallocate
        μ_ex_species = zeros(Ns)
        μ_ex_species_neut = zeros(Ns)
        μ_ex_species_coul = zeros(Ns)

        for (iλ, λv) in enumerate(λ_values)
            #println(iλ," ",λv)
            #pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss, 
            #                   ϵ_gauss2, σ_gauss2, μ_gauss2, 
            #                   ϵ_gauss3, σ_gauss3, μ_gauss3, 
            #                   ϵ_gauss4, σ_gauss4, μ_gauss4, Z, false, λ=λv)
            pref = B / σ / T   # matches earlier kB*B/σ*Sones/kBT
            #hs = HardSpheres(σ_vec .* 1e10 .* 0.5)
            soft = InversePowerLawMixture(λv .* pref .* ϵ_vec, σ_vec ./ σ, n)
            pref_gauss = 1/pref
            g1 = Gaussian(λv .* pref_gauss .* [ϵ_gauss, ϵ_gauss], [σ_gauss, σ_gauss], [μ_gauss, μ_gauss])
            g2 = Gaussian(λv .* pref_gauss .* [ϵ_gauss2, ϵ_gauss2], [σ_gauss2, σ_gauss2], [μ_gauss2, μ_gauss2])
            g3 = Gaussian(λv .* pref_gauss .* [ϵ_gauss3, ϵ_gauss3], [σ_gauss3, σ_gauss3], [μ_gauss3, μ_gauss3])
            g4 = Gaussian(λv .* pref_gauss .* [ϵ_gauss4, ϵ_gauss4], [σ_gauss4, σ_gauss4], [μ_gauss4, μ_gauss4])
            pref_gauss = 1/pref
            #coul = CustomCoulomb(Z, λv .* bjerrum_length / σ)
            #Composite Potential
            pot = CompositePotential(soft, g1, g2, g3, g4)#, coul)
            system = SimpleMixture(dims, ρ_vec, 1, pot)
            system = SimpleChargedMixture(system, Z, λv * bjerrum_length / σ)
            if iλ == 1
                sol = solve(system, closure, imethod; coulombsplitting=EwaldSplitting(3.0))
            else
                sol = solve(system, closure, imethod, gamma_0=gamma_r; coulombsplitting=EwaldSplitting(3.0))
            end
            r = sol.r 
            gλ = sol.gr
            gamma_r = OrnsteinZernike.convert_3darr_to_vec_of_smat(sol.gamma_r)
            #dr = r[2] - r[1]
            dr_int = r[2] - r[1]
            u_neut = OrnsteinZernike.evaluate_potential.(Ref(OrnsteinZernike.base_of(system).potential), r)
            u_coul = OrnsteinZernike.evaluate_coulomb_potential(r, system) .* kBT
            # Divide by λv to get unscaled u(r) because the potential inside potential is scaled by λ
            u = u_neut .+ u_coul
            u ./= λv
            u_neut ./= λv
            u_coul ./= λv
            # Sum over unique pairs to avoid double counting
            sum_pairs = 0.0
            sum_pairs_neut = 0.0
            sum_pairs_coul = 0.0
            # Compute μ_ex for each species
            for i in 1:Ns
                sum_i = 0.0
                sum_i_neut = 0.0
                sum_i_coul = 0.0
                for j in 1:Ns
                    # evaluate i-j potential at all r
                    u_ij = getindex.(u, i, j)
                    sum_i += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_ij) * dr_int
                    u_neut_ij = getindex.(u_neut, i, j)
                    sum_i_neut += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_neut_ij) * dr_int
                    u_coul_ij = getindex.(u_coul, i, j)
                    sum_i_coul += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_coul_ij) * dr_int
                end
                μ_ex_species[i] += 4π * sum_i * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
                μ_ex_species_neut[i] += 4π * sum_i_neut * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
                μ_ex_species_coul[i] += 4π * sum_i_coul * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
            end
            #integrand_λ[iλ] = 4π * sum_pairs
            #μ_ex_J_old = μ_ex_J
            #if iλ>1
            #    dλ = λ_values[iλ] - λ_values[iλ-1]
            #    μ_ex_J += 0.5 * dλ * (integrand_λ[iλ] + integrand_λ[iλ-1])
            #end
            #err_μ_ex = abs(μ_ex_J - μ_ex_J_old)/abs(μ_ex_J_old)
            #println(iλ," ",err_μ_ex," ",μ_ex_J)
        end
        #μ_ex_J = 0.0
        #for i in 1:(nλ-1)
        #    dλ = λ_values[i+1] - λ_values[i]
        #    μ_ex_J += 0.5 * dλ * (integrand_λ[i] + integrand_λ[i+1])
        #end
        #ν = [1.0, 1.0]  # stoichiometric coefficients of Na+ and Cl-
        μ_ions = μ_ex_species
        μ_ions_neut = μ_ex_species_neut
        μ_ions_coul = μ_ex_species_coul
        ln_gamma_pm = (νc .* μ_ions[1] + νa .* μ_ions[2]) / (νc + νa)
        ln_gamma_pm_neut = (νc .* μ_ions_neut[1] + νa .* μ_ions_neut[2]) / (νc + νa)
        ln_gamma_pm_coul = (νc .* μ_ions_coul[1] + νa .* μ_ions_coul[2]) / (νc + νa)
        #println(nλ," ",ln_gamma_pm)
        println("nλ: ",nλ," | tot: ",ln_gamma_pm," | N: ",ln_gamma_pm_neut," | C: ",ln_gamma_pm_coul," | err: ",ln_gamma_pm-(ln_gamma_pm_neut+ln_gamma_pm_coul))
    end
    
    return lnγ_mean = μ_ex_J / (2 * kBT)
end

function activity_coefficient_charging(c; T=298.15, nλ=100)
    ρ0 = c * 6.02214076e23 * 1e3  
    ρ = [ρ0, ρ0]

    #pref = B / σ / T   # matches earlier kB*B/σ*Sones/kBT
    #pref_gauss = 1/pref
    #soft = InversePowerLawMixture(λv .* pref .* ϵ_vec, σ_vec ./ σ, n)
    #g1 = Gaussian(λv .* pref_gauss .* [ϵ_gauss, ϵ_gauss], [σ_gauss, σ_gauss], [μ_gauss, μ_gauss])
    #g2 = Gaussian(λv .* pref_gauss .* [ϵ_gauss2, ϵ_gauss2], [σ_gauss2, σ_gauss2], [μ_gauss2, μ_gauss2])
    #g3 = Gaussian(λv .* pref_gauss .* [ϵ_gauss3, ϵ_gauss3], [σ_gauss3, σ_gauss3], [μ_gauss3, μ_gauss3])
    #g4 = Gaussian(λv .* pref_gauss .* [ϵ_gauss4, ϵ_gauss4], [σ_gauss4, σ_gauss4], [μ_gauss4, μ_gauss4])
    #pot = CompositePotential(soft, g1, g2, g3, g4)
    empty!(POTENTIAL_CACHE)
    pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss,
                               ϵ_gauss2,  σ_gauss2, μ_gauss2,
                               ϵ_gauss3,  σ_gauss3, μ_gauss3,
                               ϵ_gauss4,  σ_gauss4, μ_gauss4, Z, false)
    system = SimpleMixture(dims, ρ_vec, 1, pot)
    system = SimpleChargedMixture(system, Z, λv * bjerrum_length / σ)
    closure = HypernettedChain()
    ngmethod  = NgIteration(M = M,
                          dr = dr,
                          tolerance = 1e-6,
                          verbose = false,
                          max_iterations = 10^6,
                          N_stages = 8)

    fmethod = FourierIteration(M=M, dr=dr, 
                                tolerance=1e-4, 
                                verbose=false, 
                                max_iterations=10^6, 
                                mixing_parameter=0.01)

    #Charging route
    kBT = OrnsteinZernike.kBT_of(system)
    ρ_vec = diag(OrnsteinZernike.ρ_of(system))
    Ns = OrnsteinZernike.number_of_species(system)
    dims = OrnsteinZernike.dims_of(system)
    λ_values = [0.001 .+ (0.999) .* ((i-1)/(nλ-1))^2 for i in 1:nλ]
    integrand_λ = zeros(nλ)
    gamma_r = nothing
    
    # Preallocate
    μ_ex_species = zeros(Ns)
    #μ_ex_species_neut = zeros(Ns)
    #μ_ex_species_coul = zeros(Ns)

    for (iλ, λv) in enumerate(λ_values)
        #soft = InversePowerLawMixture(λv .* pref .* ϵ_vec, σ_vec ./ σ, n)
        #g1 = Gaussian(λv .* pref_gauss .* [ϵ_gauss, ϵ_gauss], [σ_gauss, σ_gauss], [μ_gauss, μ_gauss])
        #g2 = Gaussian(λv .* pref_gauss .* [ϵ_gauss2, ϵ_gauss2], [σ_gauss2, σ_gauss2], [μ_gauss2, μ_gauss2])
        #g3 = Gaussian(λv .* pref_gauss .* [ϵ_gauss3, ϵ_gauss3], [σ_gauss3, σ_gauss3], [μ_gauss3, μ_gauss3])
        #g4 = Gaussian(λv .* pref_gauss .* [ϵ_gauss4, ϵ_gauss4], [σ_gauss4, σ_gauss4], [μ_gauss4, μ_gauss4])
        #pot = CompositePotential(soft, g1, g2, g3, g4)
        empty!(POTENTIAL_CACHE)
        pot = get_cached_potential(σ, σ_vec, ϵ_vec, ϵ_gauss, σ_gauss, μ_gauss,
                                ϵ_gauss2,  σ_gauss2, μ_gauss2,
                                ϵ_gauss3,  σ_gauss3, μ_gauss3,
                                ϵ_gauss4,  σ_gauss4, μ_gauss4, Z, false; λ=λv)
        system = SimpleMixture(dims, ρ_vec, 1, pot)
        system = SimpleChargedMixture(system, Z, λv * bjerrum_length / σ)
        if iλ == 1
            try
                sol = solve(system, closure, ngmethod; coulombsplitting=EwaldSplitting(3.0))
            catch
                sol = solve(system, closure, fmethod; coulombsplitting=EwaldSplitting(3.0))
            end
        else
            try
                sol = solve(system, closure, ngmethod, gamma_0=gamma_r; coulombsplitting=EwaldSplitting(3.0))
            catch
                sol = solve(system, closure, fmethod; coulombsplitting=EwaldSplitting(3.0))
            end
        end
        r = sol.r 
        gλ = sol.gr
        gamma_r = OrnsteinZernike.convert_3darr_to_vec_of_smat(sol.gamma_r)

        dr_int = r[2] - r[1]
        u_neut = OrnsteinZernike.evaluate_potential.(Ref(OrnsteinZernike.base_of(system).potential), r)
        u_coul = OrnsteinZernike.evaluate_coulomb_potential(r, system) .* kBT

        # Divide by λv to get unscaled u(r) because the potential inside potential is scaled by λ
        u = u_neut .+ u_coul
        u ./= λv
        #u_neut ./= λv
        #u_coul ./= λv
        # Sum over unique pairs to avoid double counting
        sum_pairs = 0.0
        #sum_pairs_neut = 0.0
        #sum_pairs_coul = 0.0
        # Compute μ_ex for each species
        for i in 1:Ns
            sum_i = 0.0
            #sum_i_neut = 0.0
            #sum_i_coul = 0.0
            for j in 1:Ns
                # evaluate i-j potential at all r
                u_ij = getindex.(u, i, j)
                sum_i += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_ij) * dr_int
                #u_neut_ij = getindex.(u_neut, i, j)
                #sum_i_neut += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_neut_ij) * dr_int
                #u_coul_ij = getindex.(u_coul, i, j)
                #sum_i_coul += ρ_vec[j] * sum(r.^2 .* gλ[:, i, j] .* u_coul_ij) * dr_int
            end
            μ_ex_species[i] += 4π * sum_i * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
            #μ_ex_species_neut[i] += 4π * sum_i_neut * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
            #μ_ex_species_coul[i] += 4π * sum_i_coul * (iλ == 1 ? 0.0 : (λ_values[iλ] - λ_values[iλ-1]))
        end
    end
    μ_ions = μ_ex_species
    #μ_ions_neut = μ_ex_species_neut
    #μ_ions_coul = μ_ex_species_coul
    ln_gamma_pm = (νc .* μ_ions[1] + νa .* μ_ions[2]) / (νc + νa)
    #ln_gamma_pm_neut = (νc .* μ_ions_neut[1] + νa .* μ_ions_neut[2]) / (νc + νa)
    #ln_gamma_pm_coul = (νc .* μ_ions_coul[1] + νa .* μ_ions_coul[2]) / (νc + νa)
    
    return ln_gamma_pm
end
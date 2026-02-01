function calc_inconsistency_ρ(system, dims, ρ, σ, pot, Z, bjerrum_length, kBT, closure, imethod)
    system = SimpleMixture(dims, ρ*σ^3, 1, pot)
    system = SimpleChargedMixture(system, Z, bjerrum_length / σ)
    sol = solve(system, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
    p1 = compute_virial_pressure_charged(sol, system)
    println("p1: ",p1)
    
    dρ = ρ.*1e-4#sqrt.(eps.(ρ))
    system2 = SimpleMixture(dims, (ρ+dρ)*σ^3, 1, pot)
    system2 = SimpleChargedMixture(system2, Z, bjerrum_length / σ)
    sol2 = solve(system2, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
    p2 = compute_virial_pressure_charged(sol2, system2)
    println("p2: ",p2)
    dpdρ = (p2-p1)/dρ
    χ = compute_compressibility_charged(sol, system)
    println("χ: ",χ)
    readline()
    inconsistency = dpdρ/kBT - 1/(ρ*kBT*χ)
    return inconsistency
end

function calc_inconsistency_ρ_ex(system, dims, ρ, σ, pot, Z, bjerrum_length, kBT, closure, imethod)
    h = 1e-3
    dρ = ρ.*h#sqrt.(eps.(ρ))
    #dρ = sqrt.(eps.(ρ))
    dkBT = 1e-3
    kBT = 1.0

    dρ_OZ = dρ .* σ.^3
    ρ_OZ = ρ .* σ.^3

    closure = ExtendedRogersYoung(α,a)
    
    system0 = SimpleMixture(dims, (ρ-dρ)*σ^3, 1, pot)
    system0 = SimpleChargedMixture(system0, Z, bjerrum_length / σ)
    sol0 = solve(system0, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
    
    system = SimpleMixture(dims, ρ*σ^3, 1, pot)
    system = SimpleChargedMixture(system, Z, bjerrum_length / σ)
    sol = solve(system, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
    
    system2 = SimpleMixture(dims, (ρ+dρ)*σ^3, 1, pot)
    system2 = SimpleChargedMixture(system2, Z, bjerrum_length / σ)
    sol2 = solve(system2, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))

    system3 = SimpleMixture(dims, ρ*σ^3, 1+dkBT, pot)
    system3 = SimpleChargedMixture(system3, Z, bjerrum_length / σ)
    sol3 = solve(system3, closure, imethod,
            coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))

    p0 = compute_virial_pressure_charged(sol0, system0)
    p2 = compute_virial_pressure_charged(sol2, system2)

    ρU0 = (ρ_OZ-dρ_OZ)*compute_excess_energy(sol0, system0)
    ρU1 = ρ_OZ*compute_excess_energy(sol, system)
    ρU2 = (ρ_OZ+dρ_OZ)*compute_excess_energy(sol2, system2)

    dpdρ = (p2-p0)/(2dρ_OZ)
    d2ρUdρ2 = (ρU2 + ρU0 - 2ρU1)/(dρ_OZ.^2)

    ĉ0_1 = OrnsteinZernike.get_ĉ0(sol, system)
    ĉ0_3 = OrnsteinZernike.get_ĉ0(sol3, system3)
    dĉ0dkBT = (ĉ0_3-ĉ0_1)/dkBT
    dĉ0dβ = - kBT^2 * dĉ0dkBT
    
    χ = compute_compressibility_charged(sol, system)
    inconsistency1 = dpdρ/kBT - 1/(ρ_OZ*kBT*χ)
    inconsistency2 = d2ρUdρ2 + dĉ0dβ
    #inconsistency = sum.([inconsistency1, inconsistency2].^2)
    inconsistency = sum.(abs.([norm(inconsistency1), norm(inconsistency2)]))
    @show inconsistency1, inconsistency2, α, a

    return inconsistency
end

function find_self_consistent_solution(ρ, kBT, method, dims, pot; lims=(0.1, 2.0))

    function RY_inconsistency(ρ, α)
        system1 = SimpleFluid(dims, ρ, kBT, pot)
        sol1 = solve(system1, RogersYoung(α), method)
        p1 = compute_virial_pressure(sol1, system1)

        dρ = sqrt(eps(ρ))
        system2 = SimpleFluid(dims, ρ+dρ, kBT, pot)
        sol2 = solve(system2, RogersYoung(α), method)
        p2 = compute_virial_pressure(sol2, system2)
        dpdρ = (p2-p1)/dρ

        χ = compute_compressibility(sol1, system1)
        inconsistency = dpdρ/kBT - 1/(ρ*kBT*χ)
        return inconsistency
    end

    func = α ->  RY_inconsistency(ρ, α)
    α =  Roots.find_zero(func, lims, Roots.Bisection(), atol=0.0001)
    system = SimpleFluid(dims, ρ, kBT, pot)
    sol = solve(system, RogersYoung(α), method)
    return system, sol, α
end


function find_self_consistent_solution_ERY(ρ, kBT, method, dims, pot; x0=[0.2, 0.2])
    function ERY_inconsistency(ρ, α2, a2)
        h = 1e-3
        dρ = ρ.*h#sqrt.(eps.(ρ))
        #dρ = sqrt.(eps.(ρ))
        dkBT = 1e-3
        kBT = 1.0

        dρ_OZ = dρ .* σ.^3
        ρ_OZ = ρ .* σ.^3

        α = sqrt(α2)
        a = sqrt(a2)

        closure = ExtendedRogersYoung(α,a)
        inconsistency = 20.0
        inconsistency1 = 20.0
        try
                system0 = SimpleMixture(dims, (ρ-dρ)*σ^3, 1, pot)
                system0 = SimpleChargedMixture(system0, Z, bjerrum_length / σ)
                sol0 = solve(system0, closure, imethod,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                system = SimpleMixture(dims, ρ*σ^3, 1, pot)
                system = SimpleChargedMixture(system, Z, bjerrum_length / σ)
                sol = solve(system, closure, imethod,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))

                system2 = SimpleMixture(dims, (ρ+dρ)*σ^3, 1, pot)
                system2 = SimpleChargedMixture(system2, Z, bjerrum_length / σ)
                sol2 = solve(system2, closure, imethod,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                system3 = SimpleMixture(dims, ρ*σ^3, 1+dkBT, pot)
                system3 = SimpleChargedMixture(system3, Z, bjerrum_length / σ)
                sol3 = solve(system3, closure, imethod,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                p0 = compute_virial_pressure_charged(sol0, system0)
                p2 = compute_virial_pressure_charged(sol2, system2)

                ρU0 = (ρ_OZ-dρ_OZ)*compute_excess_energy(sol0, system0)
                ρU1 = ρ_OZ*compute_excess_energy(sol, system)
                ρU2 = (ρ_OZ+dρ_OZ)*compute_excess_energy(sol2, system2)
                
                dpdρ = (p2-p0)/(2dρ_OZ)
                d2ρUdρ2 = (ρU2 + ρU0 - 2ρU1)/(dρ_OZ.^2)

                ĉ0_1 = OrnsteinZernike.get_ĉ0(sol, system)
                ĉ0_3 = OrnsteinZernike.get_ĉ0(sol3, system3)
                dĉ0dkBT = (ĉ0_3-ĉ0_1)/dkBT
                dĉ0dβ = - kBT^2 * dĉ0dkBT
                
                χ = compute_compressibility_charged(sol, system)
                inconsistency1 = abs.(dpdρ/kBT - 1/(ρ_OZ*kBT*χ))
                inconsistency2 = abs.(d2ρUdρ2 + dĉ0dβ)
                #inconsistency = sum.([inconsistency1, inconsistency2].^2)
                inconsistency = sum(abs.([norm(inconsistency1), norm(inconsistency2)]))
                println(sum(inconsistency)," ",sum(inconsistency1)," ",sum(inconsistency2)," ",α," ",a)
        catch
                inconsistency = 20.0
                inconsistency1 = inconsistency
                println(sum(inconsistency)," ",α," ",a)
        end
        return sum(abs.(inconsistency1))
    end

    func = x ->  ERY_inconsistency(ρ, x[1]^2, x[2]^2)
    α =  optimize(func, [x0...], NelderMead(), Optim.Options(x_tol=0.0001, f_tol=10^-6)).minimizer
    system = SimpleFluid(dims, ρ, kBT, pot)
    sol = solve(system, ExtendedRogersYoung(α[1], α[2]), method)
    return system, sol, α
end


function find_self_consistent_solution_RY(ρ, kBT, method, dims, pot; x0=[0.2])
    function RY_inconsistency(ρ, α2)
        h = 1e-3
        dρ = ρ.*h#sqrt.(eps.(ρ))
        #dρ = sqrt.(eps.(ρ))
        dkBT = 1e-3
        kBT = 1.0

        dρ_OZ = dρ .* σ.^3
        ρ_OZ = ρ .* σ.^3

        α = sqrt(α2)

        closure = ExtendedRogersYoung(α,1.0)
        closure = ZerahHansen(α)
        #inconsistency = 20.0
        inconsistency1 = 20.0
        try
                system0 = SimpleMixture(dims, (ρ-dρ)*σ^3, 1, pot)
                system0 = SimpleChargedMixture(system0, Z, bjerrum_length / σ)
                sol0 = solve(system0, closure, method,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                #system = SimpleMixture(dims, ρ*σ^3, 1, pot)
                #system = SimpleChargedMixture(system, Z, bjerrum_length / σ)
                #sol = solve(system, closure, method,
                #        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))

                system2 = SimpleMixture(dims, (ρ+dρ)*σ^3, 1, pot)
                system2 = SimpleChargedMixture(system2, Z, bjerrum_length / σ)
                sol2 = solve(system2, closure, method,
                        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                #system3 = SimpleMixture(dims, ρ*σ^3, 1+dkBT, pot)
                #system3 = SimpleChargedMixture(system3, Z, bjerrum_length / σ)
                #sol3 = solve(system3, closure, method,
                #        coulombsplitting = OrnsteinZernike.EwaldSplitting(3.0))
                
                p0 = compute_virial_pressure_charged(sol0, system0)
                p2 = compute_virial_pressure_charged(sol2, system2)

                #ρU0 = (ρ_OZ-dρ_OZ)*compute_excess_energy(sol0, system0)
                #ρU1 = ρ_OZ*compute_excess_energy(sol, system)
                #ρU2 = (ρ_OZ+dρ_OZ)*compute_excess_energy(sol2, system2)
                
                dpdρ = (p2-p0)/(2dρ_OZ)
                #d2ρUdρ2 = (ρU2 + ρU0 - 2ρU1)/(dρ_OZ.^2)

                #ĉ0_1 = OrnsteinZernike.get_ĉ0(sol, system)
                #ĉ0_3 = OrnsteinZernike.get_ĉ0(sol3, system3)
                #dĉ0dkBT = (ĉ0_3-ĉ0_1)/dkBT
                #dĉ0dβ = - kBT^2 * dĉ0dkBT
                
                χ = compute_compressibility_charged(sol, system)
                inconsistency1 = abs.(dpdρ/kBT - 1/(ρ_OZ*kBT*χ))
                #inconsistency2 = abs.(d2ρUdρ2 + dĉ0dβ)
                #inconsistency = sum(abs.([norm(inconsistency1), norm(inconsistency2)]))
                println(sum(inconsistency1)," ",α)
        catch
                inconsistency = 20.0
                inconsistency1 = inconsistency
                println(sum(inconsistency)," ",α)
        end
        return sum(abs.(inconsistency1))
    end

    func = x ->  RY_inconsistency(ρ, x[1]^2)
    α =  optimize(func, [x0...], NelderMead(), Optim.Options(x_tol=0.0001, f_tol=10^-6)).minimizer
    return α
end
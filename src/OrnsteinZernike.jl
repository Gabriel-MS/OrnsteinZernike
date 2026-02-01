
"""
A generic solver package for Ornstein-Zernike equations from liquid state theory

"""
module OrnsteinZernike
using FFTW, StaticArrays, LinearAlgebra, Hankel, SpecialFunctions, Dierckx 
using Bessels: besselj
using FunctionZeros
using Roots: find_zero
#using Optim

export solve
export SimpleFluid, SimpleMixture, SimpleChargedFluid, SimpleChargedMixture
export OZSolution
export Exact, FourierIteration, NgIteration, DensityRamp, TemperatureRamp
export PercusYevick,  HypernettedChain, MeanSpherical, ModifiedHypernettedChain, Verlet, MartynovSarkisov
export SMSA, RogersYoung, ZerahHansen, DuhHaymet, Lee, ChoudhuryGhosh, BallonePastoreGalliGazzillo
export VompeMartynov, CharpentierJackse, BomontBretonnet, Khanpour, ModifiedVerlet, CarbajalTinoko, ExtendedRogersYoung
export CustomPotential, InversePowerLaw, HardSpheres, LennardJones, SquareWell, Morse, TabulatedPotential, Yukawa, GaussianCore
export InversePowerLawMixture, Gaussian, WCAMixture, MieMixture
export CompositePotential, CustomCoulomb
export compute_compressibility, compute_excess_energy, compute_virial_pressure, compute_activity_coefficient
export evaluate_potential, evaluate_potential_derivative, discontinuities
export compute_virial_pressure_charged, compute_compressibility_charged
export WCADivision, AllShortRangeDivision, dispersion_tail
export CoulombSplitting, NoCoulombSplitting, EwaldSplitting
export calc_inconsistency_ρ, calc_inconsistency_ρ_ex
export compute_activity_coefficient_fast, calculate_miac_curve_optimized, simpson_weights


include("Systems.jl")
include("Potentials.jl")
include("Coulomb.jl")
include("PotentialSplitting.jl")
include("Closures.jl")
include("FourierTransforms.jl")
include("Solutions.jl")
include("Solvers.jl")
include("Thermodynamics.jl")
include("Consistency.jl")

end # module OrnsteinZernike

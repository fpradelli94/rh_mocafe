"""
File containing the FEniCS Forms used throughout the simulation
"""
import fenics
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters


def angiogenic_factors_form_dt(af: fenics.Function,
                               af_old: fenics.Function,
                               phi: fenics.Function,
                               c: fenics.Function,
                               v: fenics.TestFunction,
                               par: Parameters):
    """
    Time variant angiogenic factors form.
    """
    form = time_derivative_form(af, af_old, v, par) + angiogenic_factors_form_eq(af, phi, c, v, par)
    return form


def angiogenic_factors_form_eq(af: fenics.Function,
                               phi: fenics.Function,
                               c: fenics.Function,
                               v: fenics.TestFunction,
                               par: Parameters):
    """
    Equilibrium angiogenic factors form.
    """
    # load parameters
    D_af = par.get_value("D_af")
    V_pT_af = par.get_value("V_pT_af")
    V_uc_af = par.get_value("V_uc_af")
    V_d_af = par.get_value("V_d_af")
    form = \
        (D_af * fenics.dot(fenics.grad(af), fenics.grad(v)) * fenics.dx) - \
        (V_pT_af * phi * (fenics.Constant(1.) - shf(c)) * v * fenics.dx) + \
        (V_uc_af * shf(c) * af * v * fenics.dx) + \
        (V_d_af * af * v * fenics.dx)
    return form


def time_derivative_form(var: fenics.Function,
                         var_old: fenics.Function,
                         v: fenics.TestFunction,
                         par: Parameters):
    """
    General time derivative form.
    """
    return ((var - var_old) / par.get_value("dt")) * v * fenics.dx

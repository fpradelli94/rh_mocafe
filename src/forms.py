"""
File containing the FEniCS Forms used throughout the simulation
"""
import fenics
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters, _unpack_parameters_list


def angiogenic_factors_form_dt(af: fenics.Function,
                               af_old: fenics.Function,
                               phi: fenics.Function,
                               c: fenics.Function,
                               v: fenics.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Time variant angiogenic factors form.
    """
    form = time_derivative_form(af, af_old, v, par, **kwargs) + angiogenic_factors_form_eq(af, phi, c, v, par, **kwargs)
    return form


def angiogenic_factors_form_eq(af: fenics.Function,
                               phi: fenics.Function,
                               c: fenics.Function,
                               v: fenics.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Equilibrium angiogenic factors form.
    """
    # load parameters
    D_af, V_pT_af, V_uc_af, V_d_af = _unpack_parameters_list(["D_af", "V_pT_af", "V_uc_af", "V_d_af"],
                                                             par, kwargs)
    form = \
        (D_af * fenics.dot(fenics.grad(af), fenics.grad(v)) * fenics.dx) - \
        (V_pT_af * phi * (fenics.Constant(1.) - shf(c)) * v * fenics.dx) + \
        (V_uc_af * shf(c) * af * v * fenics.dx) + \
        (V_d_af * af * v * fenics.dx)
    return form


def time_derivative_form(var: fenics.Function,
                         var_old: fenics.Function,
                         v: fenics.TestFunction,
                         par: Parameters,
                         **kwargs):
    """
    General time derivative form.
    """
    dt, = _unpack_parameters_list(["dt"], par, kwargs)
    return ((var - var_old) / dt) * v * fenics.dx

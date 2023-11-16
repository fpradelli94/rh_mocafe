"""
File containing the FEniCS Forms used throughout the simulation
"""
import ufl
from dolfinx.fem import Function
from mocafe.math import shf
from mocafe.fenut.parameters import Parameters, _unpack_parameters_list


def angiogenic_factors_form_dt(af: Function,
                               af_old: Function,
                               phi: Function,
                               c: Function,
                               v: ufl.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Time variant angiogenic factors form.
    """
    form = time_derivative_form(af, af_old, v, par, **kwargs) + angiogenic_factors_form_eq(af, phi, c, v, par, **kwargs)
    return form


def angiogenic_factors_form_eq(af: Function,
                               phi: Function,
                               c: Function,
                               v: ufl.TestFunction,
                               par: Parameters,
                               **kwargs):
    """
    Equilibrium angiogenic factors form.
    """
    # load parameters
    D_af, V_pT_af, V_uc_af, V_d_af = _unpack_parameters_list(["D_af", "V_pT_af", "V_uc_af", "V_d_af"],
                                                             par, kwargs)
    form = \
        (D_af * ufl.dot(ufl.grad(af), ufl.grad(v)) * ufl.dx) - \
        (V_pT_af * phi * (1 - shf(c)) * v * ufl.dx) + \
        (V_uc_af * shf(c) * af * v * ufl.dx) + \
        (V_d_af * af * v * ufl.dx)
    return form


def time_derivative_form(var: Function,
                         var_old: Function,
                         v: ufl.TestFunction,
                         par: Parameters,
                         **kwargs):
    """
    General time derivative form.
    """
    dt, = _unpack_parameters_list(["dt"], par, kwargs)
    return ((var - var_old) / dt) * v * ufl.dx

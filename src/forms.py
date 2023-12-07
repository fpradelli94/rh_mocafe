"""
File containing the FEniCS Forms used throughout the simulation
"""
import logging
import ufl
import dolfinx
from dolfinx.fem import Function, Constant
from mocafe.fenut.parameters import Parameters, _unpack_parameters_list

logger = logging.getLogger(__name__)


def shf(mesh: dolfinx.mesh.Mesh, variable, slope: float = 100):
    r"""
    Smoothed Heavyside Function (SHF) using the sigmoid function, which reads:

    .. math::
        \frac{e^{slope * variable}}{(1 + e^{slope * variable})}


    :param mesh: domain (necessary for constant)
    :param variable: varible for the SHF
    :param slope: slope of the SHF. Default is 100
    :return: the value of the sigmoid for the given value of the variable
    """
    slope = Constant(mesh, dolfinx.default_scalar_type(slope))
    return ufl.exp(slope * variable) / (Constant(mesh, dolfinx.default_scalar_type(1.)) + ufl.exp(slope * variable))


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
    # transform in constants
    mesh = phi.function_space.mesh
    D_af = Constant(mesh, dolfinx.default_scalar_type(D_af))
    V_pT_af = Constant(mesh, dolfinx.default_scalar_type(V_pT_af))
    V_uc_af = Constant(mesh, dolfinx.default_scalar_type(V_uc_af))
    V_d_af = Constant(mesh, dolfinx.default_scalar_type(V_d_af))

    # define each part of the form
    # diffusion
    diffusion_form = (D_af * ufl.dot(ufl.grad(af), ufl.grad(v)) * ufl.dx)
    # production
    production_term = V_pT_af * phi * (Constant(mesh, dolfinx.default_scalar_type(1.)) - shf(mesh, c))
    production_term_non_negative = ufl.conditional(
        condition=ufl.gt(production_term, Constant(mesh, dolfinx.default_scalar_type(0.))),
        true_value=production_term,
        false_value=Constant(mesh, dolfinx.default_scalar_type(0.))
    )
    production_form = production_term_non_negative * v * ufl.dx
    # uptake
    uptake_term = V_uc_af * shf(mesh, c)
    uptake_term_non_negative = ufl.conditional(
        condition=ufl.gt(uptake_term, Constant(mesh, dolfinx.default_scalar_type(0.))),
        true_value=uptake_term,
        false_value=Constant(mesh, dolfinx.default_scalar_type(0.))
    )
    uptake_form = uptake_term_non_negative * af * v * ufl.dx
    # degradation
    degradation_form = V_d_af * af * v * ufl.dx

    # assemble form
    form = diffusion_form - production_form + uptake_form + degradation_form

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
    if isinstance(dt, int) or isinstance(dt, float):
        dt = Constant(var_old.function_space.mesh, dolfinx.default_scalar_type(dt))
    return ((var - var_old) / dt) * v * ufl.dx


def angiogenesis_form_no_proliferation(c: Function,
                                       c0: Function,
                                       mu: Function,
                                       mu0: Function,
                                       v1: ufl.TestFunction,
                                       v2: ufl.TestFunction,
                                       parameters: Parameters = None,
                                       **kwargs):
    r"""
    (New in version 1.4)
    Returns the UFL form for the Phase-Field model for angiogenesis reported by Travasso et al. (2011)
    :cite:`Travasso2011a`, without the proliferation term.

    The equation reads simply as:

    .. math::
       \frac{\partial c}{\partial t} = M \cdot \nabla^2 [\frac{df}{dc}\ - \epsilon \nabla^2 c]

    Where :math: `c` is the unknown field representing the capillaries, and :

    .. math:: f = \frac{1}{4} \cdot c^4 - \frac{1}{2} \cdot c^2

    In this implementation, the equation is split in two equations of lower order, in order to make the weak form
    solvable using standard Lagrange finite elements:

    .. math::
       \frac{\partial c}{\partial t} &= M \nabla^2 \cdot \mu\\
       \mu &= \frac{d f}{d c} - \epsilon \nabla^{2}c

    Specify a parameter for the form calling the function, e.g. with
    ``angiogenesis_form(c, c0, mu, mu0, v1, v2, af, parameters, alpha_p=10, M=20)``. If both a Parameters object and a
    parameter as input are given, the function will choose the input parameter.

    :param c: capillaries field
    :param c0: initial condition for the capillaries field
    :param mu: auxiliary field
    :param mu0: initial condition for the auxiliary field
    :param v1: test function for c
    :param v2: test function  for mu
    :param parameters: simulation parameters
    :return:
    """
    # get parameters
    dt, epsilon, M = _unpack_parameters_list(["dt", "epsilon", "M"],
                                             parameters,
                                             kwargs)
    # define theta
    theta = 0.5

    # define chemical potential for the phase field
    c = ufl.variable(c)
    four = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(4))
    two = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(2))
    chem_potential = ((c ** four) / four) - ((c ** two) / two)

    # define total form
    form_cahn_hillard = cahn_hillard_form(c, c0, mu, mu0, v1, v2, dt, theta, chem_potential,
                                          epsilon, M)
    return form_cahn_hillard


def cahn_hillard_form(c,
                      c0: Function,
                      mu: Function,
                      mu0: Function,
                      q: Function,
                      v: Function,
                      dt,
                      theta,
                      chem_potential,
                      lmbda,
                      M):
    r"""
    Returns the UFL form of a for a general Cahn-Hillard equation, discretized in time using the theta method. The
    method is the same reported by the FEniCS team in one of their demo `1. Cahn-Hillard equation`_ and is briefly
    discussed below for your conveneince.

    .. _1. Cahn-Hillard equation:
       https://fenicsproject.org/olddocs/dolfin/2016.2.0/cpp/demo/documented/cahn-hilliard/cpp/documentation.html

    The Cahn-Hillard equation reads as follows:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M (\nabla(\frac{d f}{d c}
             - \lambda \nabla^{2}c)) = 0 \quad \textrm{in} \ \Omega

    Where :math: `c` is the unknown field to find, :math: `f` is some kind of energetic potential which defines the
    phase separation, and :math: `M` is a scalar parameter.

    The equation involves 4th order derivatives, so its weak form could not be handled with the standard Lagrange
    finite element basis. However, the equation can be split in two second-order equations adding a second unknown
    auxiliary field :math: `\mu`:

    .. math::
       \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad \textrm{in} \ \Omega, \\
       \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad \textrm{ in} \ \Omega.

    In this way, it is possible to solve this equation using the standard Lagrange basis and, indeed, this
    implementation uses this form.

    :param c: main Cahn-Hillard field
    :param c0: initial condition for the main Cahn-Hillard field
    :param mu: auxiliary field for the Cahn-Hillard equation
    :param mu0: initial condition for the auxiliary field
    :param q: test function for c
    :param v: test function for mu
    :param dt: time step
    :param theta: theta value for theta method
    :param chem_potential: UFL form for the Cahn-Hillard potential
    :param lmbda: energetic weight for the gradient of c
    :param M: scalar parameter
    :return: the UFL form of the Cahn-Hillard Equation
    """
    # Define form for mu (theta method)
    theta = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(theta))
    mu_mid = (Constant(c0.function_space.mesh, dolfinx.default_scalar_type(1.)) - theta) * mu0 + theta * mu

    # chem potential derivative
    dfdc = ufl.diff(chem_potential, c)

    # constants
    if isinstance(dt, int) or isinstance(dt, float):
        dt = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(dt))
    if isinstance(M, float):
        M = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(M))
    lmbda = Constant(c0.function_space.mesh, dolfinx.default_scalar_type(lmbda))

    # define form
    l0 = (c - c0) * q * ufl.dx + dt * M * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
    l1 = mu * v * ufl.dx - dfdc * v * ufl.dx - lmbda * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx
    form = l0 + l1

    # return form
    return form


def chan_hillard_free_enery(mu: Function):

    energy = - (ufl.grad(mu) ** 2) * ufl.dx

    return energy

from __future__ import division

from simtbx.diffBragg.refiners.crystal_systems import CrystalSystemManager
import numpy as np
from scitbx.matrix import sqr


class TriclinicManager(CrystalSystemManager):
    """
    A class to manage the properties and derivatives of a triclinic crystal system.
    In this system, a, b, c, alpha, beta, and gamma are all independent variables.
    """

    def __init__(self, a=10, b=11, c=12, alpha=70*np.pi/180, beta=80*np.pi/180, gamma=90*np.pi/180):
        """
        Initializes the triclinic manager.
        Parameters are converted to Angstroms and radians for internal calculations.
        :param a: unit cell a parameter in Angstroms
        :param b: unit cell b parameter in Angstroms
        :param c: unit cell c parameter in Angstroms
        :param alpha: unit cell alpha angle in radians 
        :param beta: unit cell beta angle in radians
        :param gamma: unit cell gamma angle in radians
        """
        self.variables = [a, b, c, alpha, beta, gamma]

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, val):
        self._variables = val

    @property
    def derivative_matrices(self):
        """Returns a list of the first derivative matrices of B with respect to each variable."""
        return [self._dB_da_real, self._dB_db_real, self._dB_dc_real,
                self._dB_dalpha_real, self._dB_dbeta_real, self._dB_dgamma_real]

    @property
    def second_derivative_matrices(self):
        """
        Returns a list of the second derivative matrices.
        NOTE: Some complex terms, especially for the B_33 element, are set to 0.
        """
        return [self._d2B_da2_real, self._d2B_db2_real, self._d2B_dc2_real,
                self._d2B_dalpha2_real, self._d2B_dbeta2_real, self._d2B_dgamma2_real]

    # Unit cell parameter properties
    @property
    def a(self):
        return self.variables[0]

    @property
    def b(self):
        return self.variables[1]

    @property
    def c(self):
        return self.variables[2]

    @property
    def al(self):
        return self.variables[3]

    @property
    def be(self):
        return self.variables[4]

    @property
    def ga(self):
        return self.variables[5]

    @property
    def variable_names(self):
        """Returns the names of the variables."""
        return self._names

    # First derivative properties
    @property
    def _dB_da_real(self):
        """Derivative of B matrix with respect to a."""
        return sqr((1, 0, 0,
                    0, 0, 0,
                    0, 0, 0))

    @property
    def _dB_db_real(self):
        """Derivative of B matrix with respect to b."""
        return sqr((0, self.cga, 0,
                    0, self.sga, 0,
                    0, 0, 0))

    @property
    def _dB_dc_real(self):
        """Derivative of B matrix with respect to c."""
        # V is proportional to c, so d(V)/dc = V/c
        dB33_dc = self.V / (self.a * self.b * self.c * self.sga)
        return sqr((0, 0, self.cbe,
                    0, 0, (self.cal - self.cbe * self.cga) / self.sga,
                    0, 0, dB33_dc))

    @property
    def _dB_dalpha_real(self):
        """Derivative of B matrix with respect to alpha."""
        Omega = self.V / (self.a * self.b * self.c)
        if Omega == 0: return sqr((0,)*9)
        dB23_dal = -self.c * self.sal / self.sga
        dB33_dal = (self.c * self.sal * (self.cal - self.cbe * self.cga)) / (self.sga * Omega)
        return sqr((0, 0, 0,
                    0, 0, dB23_dal,
                    0, 0, dB33_dal))

    @property
    def _dB_dbeta_real(self):
        """Derivative of B matrix with respect to beta."""
        Omega = self.V / (self.a * self.b * self.c)
        if Omega == 0: return sqr((0,)*9)
        dB13_dbe = -self.c * self.sbe
        dB23_dbe = self.c * self.sbe * self.cga / self.sga
        dB33_dbe = (self.c * self.sbe * (self.cbe - self.cal * self.cga)) / (self.sga * Omega)
        return sqr((0, 0, dB13_dbe,
                    0, 0, dB23_dbe,
                    0, 0, dB33_dbe))

    @property
    def _dB_dgamma_real(self):
        """Derivative of B matrix with respect to gamma."""
        Omega = self.V / (self.a * self.b * self.c)
        if Omega == 0: return sqr((0,)*9)
        dB12_dga = -self.b * self.sga
        dB22_dga = self.b * self.cga
        dB23_dga = (self.c * self.cbe - self.c * self.cal * self.cga) / self.sga ** 2

        # Derivative of V with respect to gamma
        dV_dga = self.a * self.b * self.c * self.sga * (self.cga - self.cal * self.cbe) / Omega
        # Use product rule for d(V/sga)/dga
        dB33_dga = (1 / (self.a * self.b)) * (dV_dga / self.sga + self.V * (-self.cga / self.sga ** 2))

        return sqr((0, dB12_dga, 0,
                    0, dB22_dga, dB23_dga,
                    0, 0, dB33_dga))

    # Second derivative properties (simplified)
    @property
    def _d2B_da2_real(self):
        return sqr((0, 0, 0, 0, 0, 0, 0, 0, 0))

    @property
    def _d2B_db2_real(self):
        return sqr((0, 0, 0, 0, 0, 0, 0, 0, 0))

    @property
    def _d2B_dc2_real(self):
        return sqr((0, 0, 0, 0, 0, 0, 0, 0, 0))

    @property
    def _d2B_dalpha2_real(self):
        """Second derivative wrt alpha. Term for B_33 is complex and set to 0."""
        d2B23_dal2 = -self.c * self.cal / self.sga
        return sqr((0, 0, 0,
                    0, 0, d2B23_dal2,
                    0, 0, 0))

    @property
    def _d2B_dbeta2_real(self):
        """Second derivative wrt beta. Term for B_33 is complex and set to 0."""
        d2B13_dbe2 = -self.c * self.cbe
        d2B23_dbe2 = self.c * self.cbe * self.cga / self.sga
        return sqr((0, 0, d2B13_dbe2,
                    0, 0, d2B23_dbe2,
                    0, 0, 0))

    @property
    def _d2B_dgamma2_real(self):
        """Second derivative wrt gamma. Terms for B_23 and B_33 are complex and set to 0."""
        d2B12_dga2 = -self.b * self.cga
        d2B22_dga2 = -self.b * self.sga
        return sqr((0, d2B12_dga2, 0,
                    0, d2B22_dga2, 0,
                    0, 0, 0))

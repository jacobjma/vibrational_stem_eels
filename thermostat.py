import numpy as np
from ase import units
from ase.md.md import MolecularDynamics
from scipy import linalg

unit_conversions = {
    "atomic time units^-1": 1 / units.fs * 1e-15 / units._aut,
    "picoseconds^-1": 1 / 1000 / units.fs,
    "seconds^-1": 1 / units.fs * 1e-15,
    "femtoseconds^-1": units._aut / 1e-15,
    "eV": 1,
    "atomic energy units": units.Ha,
    "K": units.kB,
}


def read_gle4md_file(filename, start, stop=None):
    read = False

    matrix = []
    with open(filename) as glefile:
        for line in glefile:
            if read:
                if stop:
                    if stop in line:
                        break

                matrix += [[float(x) for x in line.split()]]

            if start in line:
                read = True
                unit_name = line.strip().split("(")[-1].replace(")", "")

    return np.array(matrix) * unit_conversions[unit_name]


class GLEThermostat(MolecularDynamics):

    def __init__(self, atoms, timestep, A=None, C=None, gle4md_file=None, trajectory=None, logfile=None,
                 loginterval=1, append_trajectory=False):

        if gle4md_file is not None:
            if (A is not None) or (C is not None):
                raise ValueError()

            A = read_gle4md_file(gle4md_file, start='# A MATRIX:', stop='# C MATRIX:')
            C = read_gle4md_file(gle4md_file, start='# C MATRIX:')

        self.T = linalg.expm(-0.5 * timestep * A)
        self.S = linalg.sqrtm(C - np.dot(self.T, np.dot(C, self.T.T)))

        self.thermostat_factor = np.sqrt(atoms.get_masses())[..., None, None]
        self.thermostat_momenta = np.random.randn(*atoms.get_momenta().shape, self.S.shape[-1])
        self.thermostat_momenta = np.dot(self.thermostat_momenta, self.S)

        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile, loginterval,
                                   append_trajectory=append_trajectory)

    def _apply_thermostat(self, p):
        self.thermostat_momenta[:, :, 0] = p

        thermostat_noise = np.random.randn(*self.thermostat_momenta.shape)

        self.thermostat_momenta = (np.dot(self.thermostat_momenta, self.T)
                                   + np.dot(thermostat_noise, self.S) * self.thermostat_factor)

        return self.thermostat_momenta[:, :, 0]

    def step(self):
        atoms = self.atoms

        p = self._apply_thermostat(atoms.get_momenta())
        p = p + 0.5 * self.dt * atoms.get_forces()

        atoms.set_positions(atoms.get_positions() + self.dt * p / atoms.get_masses()[:, None])
        atoms.set_momenta(p)

        forces = atoms.get_forces(md=True)

        p = atoms.get_momenta() + 0.5 * self.dt * forces
        atoms.set_momenta(self._apply_thermostat(p))

        return forces


class DeltaThermostat(GLEThermostat):

    def __init__(self, atoms, timestep, temperature, peak_frequency, trajectory=None, logfile=None,
                 loginterval=1, append_trajectory=False):
        gle4md_file = 'delta_thermo_10THz_300K.txt'
        A = read_gle4md_file(gle4md_file, start='# A MATRIX:', stop='# C MATRIX:')
        C = read_gle4md_file(gle4md_file, start='# C MATRIX:')
        A = A / 10 * peak_frequency
        C = C / 300 * temperature / units.kB
        super().__init__(atoms, A=A, C=C, timestep=timestep, trajectory=trajectory, logfile=logfile,
                         loginterval=loginterval, append_trajectory=append_trajectory)

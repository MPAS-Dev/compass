import matplotlib.pyplot as plt
import numpy as np

from alphaBetaLab.alphaBetaLab.abEstimateAndSave import (
    abEstimateAndSaveTriangularEtopo1,
    triMeshSpecFromMshFile,
)
# importing from alphaBetaLab the needed components
from alphaBetaLab.alphaBetaLab.abOptionManager import abOptions
from compass import Step


class WavesUostFiles(Step):
    """
    A step for creating the unresolved obstacles file for wave mesh
    """
    def __init__(self, test_case, wave_culled_mesh, wave_rotate_mesh,
                 name='uost_files', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        # other things INIT should do?
        culled_mesh_path = wave_culled_mesh.path
        self.add_input_file(
            filename='wave_mesh_culled.msh',
            work_dir_target=f'{culled_mesh_path}/wave_mesh_culled.msh')

        angled_path = wave_rotate_mesh.path
        self.add_input_file(
            filename='angled.d',
            work_dir_target=f'{angled_path}/angled.d')

        self.add_input_file(
            filename='etopo1_180.nc',
            target='etopo1_180.nc',
            database='bathymetry_database')

    def run(self):
        """
        Create unresolved obstacles for wave mesh and spectral resolution
        """

        ndir = self.config.getint('wave_mesh', 'ndir')
        nfreq = self.config.getint('wave_mesh', 'nfreq')

        dirs = np.linspace(0, 2 * np.pi, ndir)
        minfrq = .035
        if (nfreq == 50):
            frqfactor = 1.07
        elif (nfreq == 36):
            frqfactor = 1.10
        elif (nfreq == 25):
            frqfactor = 1.147
        else:
            print("ERROR: Spectral resolution not supported.")
            print("Number of wave freqencies must be 25, 36, or 50.")

        freqs = [minfrq * (frqfactor ** i) for i in range(1, nfreq + 1)]

        # definition of the spatial mesh
        gridname = 'waves_mesh_culled'  # SB NOTE: should be flexible
        mshfile = 'wave_mesh_culled.msh'
        triMeshSpec = triMeshSpecFromMshFile(mshfile)

        # path of the etopo1 bathymetry
        etopoFilePath = 'etopo1_180.nc'

        # output directory
        outputDestDir = 'output/'

        # number of cores for parallel computing
        nParWorker = 1
        #nParWorker = self.config.getint('parallel', 'cores_per_node')

        # this option indicates that the computation
        # should be skipped for cells smaller than 3 km
        minSizeKm = 3
        opt = abOptions(minSizeKm=minSizeKm)

        # instruction to do the computation and save the output
        # abEstimateAndSaveTriangularGebco(
        # dirs, freqs, gridname, triMeshSpec,
        # etopoFilePath, outputDestDir, nParWorker, abOptions=opt)
        abEstimateAndSaveTriangularEtopo1(
            dirs, freqs, gridname, triMeshSpec, etopoFilePath,
            outputDestDir, nParWorker, abOptions=opt)

        theta = np.radians(np.linspace(0.0, 360.0, ndir, endpoint=False))
        freq = np.linspace(0.0, 1.0, nfreq)
        Theta, Freq = np.meshgrid(theta, freq)

        data = np.loadtxt('angled.d')
        angled = data[:, 2]

        filename_local_in = 'obstructions_local.waves_mesh_culled.in'
        filename_local_out = 'obstructions_local.waves_mesh_culled.rtd.in'
        lon_local, lat_local, nodes, sizes, alpha_local_avg, beta_local_avg, alpha_spec, beta_spec = self.read_alpha_beta(filename_local_in, nfreq)
        alpha_interp, beta_interp = self.rotate_and_interpolate(Theta, nodes, angled, alpha_spec, beta_spec)
        header = '$WAVEWATCH III LOCAL OBSTRUCTIONS'
        self.write_alpha_beta(filename_local_out, header, nodes, lon_local, lat_local, sizes, alpha_interp, beta_interp)
        self.plot_alpha_beta_spectra(Theta, Freq, alpha_spec, beta_spec, alpha_interp, beta_interp, angled, nodes, 'local')

        filename_shadow_in = 'obstructions_shadow.waves_mesh_culled.in'
        filename_shadow_out = 'obstructions_shadow.waves_mesh_culled.rtd.in'
        lon_shadow, lat_shadow, nodes, sizes, alpha_shadow_avg, beta_shadow_avg, alpha_spec, beta_spec = self.read_alpha_beta(filename_shadow_in, nfreq)
        alpha_interp, beta_interp = self.rotate_and_interpolate(Theta, nodes, angled, alpha_spec, beta_spec)
        header = '$WAVEWATCH III SHADOW OBSTRUCTIONS'
        self.write_alpha_beta(filename_shadow_out, header, nodes, lon_shadow, lat_shadow, sizes, alpha_interp, beta_interp)
        self.plot_alpha_beta_spectra(Theta, Freq, alpha_spec, beta_spec, alpha_interp, beta_interp, angled, nodes, 'shadow')

    def write_alpha_beta(self, filename, header, nodes, lon, lat, sizes, alpha_spec, beta_spec):

        n = alpha_spec.shape[0]
        nfreq = alpha_spec.shape[1]
        ndir = alpha_spec.shape[2]

        lines = []
        lines.append(header)
        lines.append(str(n))

        for i in range(n):
            lines.append('$ ilon ilat of the cell. lon: {:.8f}, lat: {:.8f}'.format(lon[i], lat[i]))
            lines.append(str(nodes[i]) + '   1')
            lines.append(sizes[i])
            lines.append('$ mean alpha: {:.16}'.format(np.mean(alpha_spec[i, :, :])))
            lines.append('$ mean beta: {:.16}'.format(np.mean(beta_spec[i, :, :])))
            lines.append('$alpha by ik, ith')
            for j in range(nfreq):
                line = ''
                for k in range(ndir):
                    line = line + '{:.2f}  '.format(alpha_spec[i, j, k])
                lines.append(line)
            lines.append('$beta by ik, ith')
            for j in range(nfreq):
                line = ''
                for k in range(ndir):
                    line = line + '{:.2f}  '.format(beta_spec[i, j, k])
                lines.append(line)

        f = open(filename, 'w')
        for line in lines:
            f.write(line + '\n')
        f.close()

    def rotate_and_interpolate(self, Theta, nodes, angled,
                               alpha_spec, beta_spec):

        n = alpha_spec.shape[0]
        nfreq = alpha_spec.shape[1]
        ndir = alpha_spec.shape[2]

        alpha_interp = np.zeros((n, nfreq, ndir))
        beta_interp = np.zeros((n, nfreq, ndir))
        for i in range(n):
            nd = nodes[i] - 1
            Theta2 = Theta - angled[nd] * np.pi / 180.0
            for j in range(nfreq):
                alpha_interp[i, j, :] = np.interp(Theta2[j, :],
                                                  Theta[j, :],
                                                  alpha_spec[i, j, :],
                                                  period=2.0 * np.pi)
                beta_interp[i, j, :] = np.interp(Theta2[j, :],
                                                 Theta[j, :],
                                                 beta_spec[i, j, :],
                                                 period=2.0 * np.pi)

        return [alpha_interp, beta_interp]

    def plot_alpha_beta_spectra(self, Theta, Freq, alpha_spec, beta_spec,
                                alpha_interp, beta_interp, angled, nodes, kind):

        for i in range(10):
            print(i)

            fig = plt.figure(figsize=[8, 4])
            ax = fig.add_subplot(2, 2, 1, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.contourf(Theta, Freq, alpha_spec[i, :, :], 30)

            ax = fig.add_subplot(2, 2, 2, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.contourf(Theta, Freq, beta_spec[i, :, :], 30)

            ax = fig.add_subplot(2, 2, 3, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.contourf(Theta, Freq, alpha_interp[i, :, :], 30)
            ax.set_title('AnglD = ' + str(angled[nodes[i] - 1]))

            ax = fig.add_subplot(2, 2, 4, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.contourf(Theta, Freq, beta_interp[i, :, :], 30)
            ax.set_title('AnglD = ' + str(angled[nodes[i] - 1]))

            plt.savefig(kind + '_spec_' + str(i) + '.png', bbox_inches='tight')

    def read_alpha_beta(self, filename, nfreq):
        f = open(filename, 'r')
        lines = f.read().splitlines()

        nodes = []
        lon = []
        lat = []
        sizes = []
        alpha_avg = []
        beta_avg = []
        alpha_spec = []
        beta_spec = []

        line = 1  # header comment
        n = int(lines[line])
        for i in range(n):

            line = line + 1  # lon lat comment

            text = lines[line]
            text_sp = text.split()
            x = float(text_sp[7].replace(',', ''))
            y = float(text_sp[9])

            line = line + 1  # node number
            nodes.append(int(lines[line].split()[0]))

            line = line + 1  # sizes comment
            line = line + 1  # sizes
            sizes.append(lines[line])

            line = line + 1  # mean alpha
            text = lines[line]
            text_sp = text.split()
            a = float(text_sp[-1])

            line = line + 1  # mean beta
            text = lines[line]
            text_sp = text.split()
            b = float(text_sp[-1])

            line = line + 1  # alpha comment
            spectrum = []
            for i in range(nfreq):
                line = line + 1
                spectrum.append(lines[line].split())
            alpha_spec.append(spectrum)
            del spectrum

            line = line + 1  # beta comment
            spectrum = []
            for i in range(nfreq):
                line = line + 1
                spectrum.append(lines[line].split())
            beta_spec.append(spectrum)
            del spectrum

            lon.append(x)
            lat.append(y)
            alpha_avg.append(a)
            beta_avg.append(b)

        lon = np.array(lon)
        lat = np.array(lat)
        nodes = np.array(nodes)
        alpha_avg = np.array(alpha_avg)
        beta_avg = np.array(beta_avg)
        alpha_spec = np.array(alpha_spec)
        beta_spec = np.array(beta_spec)

        return [lon, lat, nodes, sizes, alpha_avg, beta_avg,
                alpha_spec, beta_spec]

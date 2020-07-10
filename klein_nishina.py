import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def KN(E_in, theta, phi, sigma_bar):
    me = 0.511; rc = 1. #electron rest mass / radius
    k0 = E_in / me
    P = 1. /  (1. + k0*(1. - np.cos(theta)))
    d_sigma = 0.5*(rc**2)*(P**2)*(P + (1/P) - 2*(np.sin(theta)**2)*(np.cos(phi)**2))
    return d_sigma / sigma_bar

def SigmaBar(E_in, theta):
    me = 0.511; rc = 1.
    k0 = E_in / me
    P = 1. / (1. + k0 * (1. - np.cos(theta)))
    k = k0 / P
    w = (1 + k) / (k ** 2)
    x = (2 * (1 + k) / (1 + 2 * k)) - (np.log(1 + 2 * k) / k)
    y = np.log(1 + 2 * k) / (2 * k)
    z = (1 + 3 * k) / ((1 + 2 * k) ** 2)
    sigma = 2 * np.pi * (rc ** 2) * (w * x + y - z)
    return np.mean(sigma)

def sample_energies(num, energy = 6., plot=False):
    samples = np.random.noncentral_chisquare(3, .000001, num) * (energy / 9)
    while samples[samples > energy].size != 0:
        greater = samples[samples > energy].shape[0]
        samples[samples > energy] = np.random.noncentral_chisquare(3, .000001, greater)
    samples[samples > energy] = energy
    if plot:
        plt.hist(samples ,bins=np.arange(0., energy, .2), density=True)
        plt.title('Ingoing Photon Spectrum - {} MV'.format(int(energy)))
        plt.xlabel('Energy')
        plt.ylabel('Relative Frequency')
        plt.show()
    return samples

def make_dataset(num_particles, energies = [6.,10.,15.,18.]):
    data = pd.DataFrame(columns=['Linac Energy', 'Theta', 'Phi', 'Cross Section'])
    for energy in energies:
        incoming = sample_energies(num_particles, energy=energy)
        phis = np.pi * np.random.random(num_particles)
        thetas = 2 * np.pi * np.random.random(num_particles) - np.pi
        sigma_bar = SigmaBar(incoming, thetas)
        outgoing = KN(incoming, thetas, phis, sigma_bar)
        df = {'Linac Energy': [energy for i in range(num_particles)],
              'Theta': thetas,
              'Phi': phis,
              'Cross Section': outgoing}
        df = pd.DataFrame(df)
        data = data.append(df, ignore_index=True)
    return data

particles = 500
energies = [6., 10., 15., 18.]#[float(i) for i in range(2, 25)]
dataset = make_dataset(particles, energies=energies)
dataset.to_csv('./compton.csv', index=False)

#Plotting
avgs = []
stds = []
it_idx = 0
for i in range(len(energies)):
    cs = dataset.iloc[it_idx:it_idx + particles,-1]
    t = np.abs(dataset.iloc[it_idx:it_idx + particles,1])
    avg = np.average(t, weights=cs)
    avgs.append(avg)
    stds.append(np.sqrt(np.average((t - avg)**2, weights=cs))/ np.sqrt(particles))
    it_idx += particles
fig, ax = plt.subplots()
plt.errorbar(energies, avgs, yerr=stds, fmt='ko')
plt.title('Forward Scattering by Energy', fontsize=14)
plt.xlabel('Linac Energy [MV]', fontsize=14)
plt.ylabel('Average Absolute Polar Angle $|\\bar{\\theta}|$', fontsize=14)
plt.savefig('forward_scattering.png', dpi=300)
plt.show()


fig, axs = plt.subplots(2)
plt.suptitle('Angular Distribution - {} MV'.format(int(dataset.iloc[0,0])), fontsize=14)
axs[0].scatter(dataset.iloc[0:500, 1], dataset.iloc[0:500, -1])
axs[0].set_xlabel('Polar Angle', fontsize=14)
axs[0].set_ylabel('$\\frac{1}{\sigma}\\frac{d\sigma}{d\Omega}$', fontsize=14)
axs[1].scatter(dataset.iloc[0:500, 2], dataset.iloc[0:500, -1])
axs[1].set_xlabel('Azimuthal Angle', fontsize=14)
axs[1].set_ylabel('$\\frac{1}{\sigma}\\frac{d\sigma}{d\Omega}$', fontsize=14)
plt.savefig('ang_dist.png', dpi=300)
plt.show()


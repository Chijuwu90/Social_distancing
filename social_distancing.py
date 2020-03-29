"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import time
from pylab import rcParams
import random


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(self, init_state=None, bounds=None, size=0.01, M=0.05, quarantine_percentage=0):
        if bounds is None:
            bounds = [-2, 2, -2, 2]
        if init_state is None:
            init_state = [[1, 0, 0, -1],
                          [-0.5, 0.5, 0.5, 0.5],
                          [-0.5, -0.5, -0.5, 0.5]]
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.sick_list = [0]
        self.sick_timing = [0]
        self.sick_last = [0]
        self.healthy_count = [100]
        self.healthy_list = []
        self.sick_count = [0]
        self.immune_list = []
        self.immune_timing = []
        self.immune_count = [0]
        self.death_list = []
        self.death_count = [0]
        self.total_time = [0]
        self.time_elapsed_factor = 1.2
        if quarantine_percentage == 0:
            self.quarantine_list = []
        else:
            self.quarantine_list = random.sample(range(1, 100), quarantine_percentage)

    def step(self, dt):

        immune_threshold_time = 100 * dt

        # if abs(max(self.total_time) - 100) < 0.1:
        #    time.sleep(10)
        # else:
        #    pass

        """step once by dt seconds"""
        self.time_elapsed = self.time_elapsed + dt

        not_moving_dots = self.death_list + self.quarantine_list
        moving_dots = list(set(np.arange(100)) - set(not_moving_dots))

        # update positions
        self.state[moving_dots, :2] = self.state[moving_dots, :2] + dt * self.state[moving_dots, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 7 * self.size)

        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # Update sick list
        contact_pairs = list(zip(ind1, ind2))
        for i in contact_pairs:

            if len(set(self.sick_list) & set(list(i))) > 0:
                new_sick = list((set(self.sick_list) | set(i)) - (set(self.sick_list) & set(i)))
                for new in new_sick:
                    if (new not in self.sick_list) & (new not in self.immune_list) & (new not in self.death_list):
                        self.sick_list = self.sick_list + [new]
                        self.sick_timing = self.sick_timing + [self.time_elapsed]
                    else:
                        pass

        self.sick_last = [self.time_elapsed - self.sick_timing[i] for i, _ in enumerate(self.sick_timing)]

        # Update sick count and total time
        self.healthy_list = set(np.arange(100)) - set(self.sick_list) - set(self.immune_list) - set(self.death_list)
        self.healthy_count.append(len(self.healthy_list))
        self.sick_count.append(len(self.sick_list))
        self.immune_count.append(len(self.immune_list))
        self.death_count.append(len(self.death_list))
        self.total_time.append(self.time_elapsed * self.time_elapsed_factor)

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2)

            # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

        immune_start_ind = self.get_start_immune(immune_threshold_time)
        if immune_start_ind is None:
            pass
        elif immune_start_ind == 0:
            new_immune = self.sick_list[0]
            self.immune_list = self.immune_list + [new_immune]
            self.sick_list = self.sick_list[1:]
            self.sick_timing = self.sick_timing[1:]
            self.sick_last = self.sick_last[1:]
        else:
            new_immune = self.sick_list[:immune_start_ind]
            self.immune_list = self.immune_list + new_immune
            self.sick_list = self.sick_list[immune_start_ind + 1:]
            self.sick_timing = self.sick_timing[immune_start_ind + 1:]
            self.sick_last = self.sick_last[immune_start_ind + 1:]

        death = self.random_death()
        if death is False:
            pass
        else:
            death_ind = random.randint(0, len(self.sick_list) - 1)
            self.death_list = self.death_list + [self.sick_list[death_ind]]
            del self.sick_list[death_ind]
            del self.sick_last[death_ind]
            del self.sick_timing[death_ind]

    def random_death(self):
        if len(self.sick_list) >= 10:
            ran = random.randint(0, 300)
            if ran <= 3:
                return True
            else:
                return False
        else:
            return False

    def get_start_immune(self, immune_threshold_time):
        if len(self.sick_last) == 0:
            return None
        elif (len(self.sick_last) == 1) & (self.sick_last[0] > immune_threshold_time):
            return 0
        else:
            for i in range(len(self.sick_last) - 1):
                if (self.sick_last[i] > immune_threshold_time) & (self.sick_last[i + 1] < immune_threshold_time):
                    return i


# ------------------------------------------------------------
# set up initial
np.random.seed(42)
n_simulate_point = 100
init_state = -0.5 + np.random.random((n_simulate_point, 4))
init_state[:, :2] *= 3.9

perc = 0
box = ParticleBox(init_state, size=0.015, quarantine_percentage=perc)
dt = 1. / 10  # 30fps

# ------------------------------------------------------------
# set up figure and animation
rcParams["figure.figsize"] = 5, 8
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

ax = fig.add_axes([0.15, 0.5, 0.8, 0.45])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title(f"Social distancing: {perc}%")
plt.xticks([])
plt.yticks([])

# hospital capacity
hospital = fig.add_axes([0.15, 0.22, 0.8, 0.25])
hospital.set_xlim(0, 100)
hospital.set_ylim(0, 100)
hospital, = hospital.plot([], [], 'k-.')

# sick particles counts
sick_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
sick_count.set_xlim(0, 100)
sick_count.set_ylim(0, 100)
plt.ylabel("N")
plt.xticks([])
sick_counts, = sick_count.plot([], [], '.', color='r', ms=2)

# healthy particles counts
healthy_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
healthy_count.set_xlim(0, 100)
healthy_count.set_ylim(0, 100)
plt.ylabel("N")
plt.xticks([])
healthy_counts, = healthy_count.plot([], [], '.', color='b', ms=2)

# immune particles counts
immune_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
immune_count.set_xlim(0, 100)
immune_count.set_ylim(0, 100)
plt.xticks([])
plt.ylabel("N")
immune_counts, = immune_count.plot([], [], '.', color='g', ms=2)

# death particles counts
death_count = fig.add_axes([0.15, 0.1, 0.8, 0.1])
death_count.set_xlim(0, 100)
death_count.set_ylim(0, 20)
plt.xlabel("∂t")
plt.ylabel("N")
death_counts, = death_count.plot([], [], '.', color='k', ms=2)

# particles holds the locations of the particles
particles, = ax.plot([], [], 'o', color='b', ms=6)

# particles holds the locations of the particles
particles_sick, = ax.plot([], [], 'o', color='r', ms=6)

particles_immune, = ax.plot([], [], 'o', color='g', ms=6)
particles_death, = ax.plot([], [], 'o', color='k', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)


def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    particles_sick.set_data([], [])
    healthy_counts.set_data([], [])
    sick_counts.set_data([], [])
    immune_counts.set_data([], [])
    death_counts.set_data([], [])
    hospital.set_data([], [])
    rect.set_edgecolor('none')
    return particles, particles_sick, healthy_counts, sick_counts, immune_counts, death_counts, hospital, rect


def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)

    particles_sick.set_data(box.state[box.sick_list, 0], box.state[box.sick_list, 1])
    particles_sick.set_markersize(ms)

    particles_immune.set_data(box.state[box.immune_list, 0], box.state[box.immune_list, 1])
    particles_immune.set_markersize(ms)

    particles_death.set_data(box.state[box.death_list, 0], box.state[box.death_list, 1])
    particles_death.set_markersize(ms)

    hospital.set_data([0, 100], [30, 30])
    healthy_counts.set_data(box.total_time, box.healthy_count)
    sick_counts.set_data(box.total_time, box.sick_count)
    immune_counts.set_data(box.total_time, box.immune_count)
    death_counts.set_data(box.total_time, box.death_count)
    return particles, particles_sick, sick_counts, particles_immune, \
           immune_counts, particles_death, death_counts, healthy_counts, hospital, rect


ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

ani.save(f'social_{perc}.mov', fps=30)
plt.show()

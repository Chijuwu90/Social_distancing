"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams
from ParticleBox import ParticleBox


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

#ani.save(f'social_{perc}.mov', fps=30)
plt.show()

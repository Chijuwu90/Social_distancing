"""
Animation of Elastic collisions with Gravity

author: Chi-Ju Wu
email: chi.ju.wu90@gmail.com
website: https://github.com/Chijuwu90/
license: BSD
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams
from ParticleBox import ParticleBox
from matplotlib.lines import Line2D


class Animate:

    def __init__(self, level=0, save=False):
        self.save = save
        self.level = level

    def run(self):

        global box, fig, dt, ax, rect, particles, particles_sick, healthy_counts, sick_counts, immune_counts, \
            death_counts, hospital, particles_immune, particles_death, sick_count, death_count

        # user-defined parameter
        n_simulate_point = 100
        isolation_percentage = self.level

        # ------------------------------------------------------------
        # set up initial
        np.random.seed(42)
        init_state = -0.5 + np.random.random((n_simulate_point, 4))
        init_state[:, :2] *= 3.9

        box = ParticleBox(init_state, size=0.012, quarantine_percentage=isolation_percentage)
        dt = 1. / 10  # 30fps

        # ------------------------------------------------------------
        # set up figure and animation
        rcParams["figure.figsize"] = 5, 8
        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax = fig.add_axes([0.15, 0.5, 0.8, 0.45])
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.title(f"{isolation_percentage}% social distancing", fontsize=10)
        plt.xticks([])
        plt.yticks([])

        # hospital capacity
        hospital = fig.add_axes([0.15, 0.22, 0.8, 0.25])
        hospital.set_xlim(0, 100)
        hospital.set_ylim(0, 100)
        hospital, = hospital.plot([], [], 'plum')

        # sick particles counts
        sick_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
        ticks = [0, 20, 40, 60, 80, 100]
        sick_count.set_yticks(ticks)

        plt.ylabel("N")
        plt.xticks([])
        sick_counts, = sick_count.plot([], [], '.', color='r', ms=2)

        # healthy particles counts
        healthy_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
        healthy_count.set_xlim(0, 100)
        healthy_count.set_ylim(0, 120)
        plt.ylabel("N")
        plt.xticks([])
        healthy_counts, = healthy_count.plot([], [], '.', color='dodgerblue', ms=2)

        # immune particles counts
        immune_count = fig.add_axes([0.15, 0.22, 0.8, 0.25])
        immune_count.set_xlim(0, 100)
        immune_count.set_ylim(0, 120)
        plt.xticks([])
        plt.ylabel("Number")
        immune_counts, = immune_count.plot([], [], '.', color='lightgreen', ms=2)

        custom_lines = [Line2D([0], [0], color="dodgerblue", lw=2),
                        Line2D([0], [0], color="r", lw=2),
                        Line2D([0], [0], color="lightgreen", lw=2),
                        Line2D([0], [0], color="plum", lw=2)]
        plt.legend(custom_lines, ['Healthy', 'Sick', 'Immune', 'Hospital Capacity'], ncol=4, fontsize=7)

        # death particles counts
        death_count = fig.add_axes([0.15, 0.08, 0.8, 0.1])
        death_count.set_xlim(0, 100)
        death_count.set_ylim(0, 15)
        plt.xlabel("Time")
        plt.ylabel("Number")
        death_counts, = death_count.plot([], [], '.', color='k', ms=2)
        custom_lines = [Line2D([0], [0], color="k", lw=2)]
        plt.legend(custom_lines, ['Dead'], fontsize=8)

        # particles holds the locations of the particles
        particles, = ax.plot([], [], 'o', color='dodgerblue', ms=6)

        # particles holds the locations of the particles
        particles_sick, = ax.plot([], [], 'o', color='r', ms=6)

        particles_immune, = ax.plot([], [], 'o', color='lightgreen', ms=6)
        particles_death, = ax.plot([], [], 'o', color='k', ms=6)

        # rect is the box edge
        rect = plt.Rectangle(box.bounds[::2],
                             box.bounds[1] - box.bounds[0],
                             box.bounds[3] - box.bounds[2],
                             ec='none', lw=2, fc='none')
        ax.add_patch(rect)

        if self.save == "True":
            while len(box.sick_list) > 0:
                ani = animation.FuncAnimation(fig, Animate.animation, frames=600,
                                              interval=10, blit=False, init_func=self.ani_setup, repeat=False)
                ani.save(f"output_animation/social_distancing_{isolation_percentage}.mov", fps=30)
                plt.show()
        else:
            while len(box.sick_list) > 0:
                ani = animation.FuncAnimation(fig, Animate.animation, frames=600,
                                              interval=10, blit=False, init_func=self.ani_setup, repeat=False)
                plt.show()



    def ani_setup(self):
        """initialize animation"""
        global box, rect, particles, particles_sick, healthy_counts, sick_counts, immune_counts, death_counts, hospital
        particles.set_data([], [])
        particles_sick.set_data([], [])
        healthy_counts.set_data([], [])
        sick_counts.set_data([], [])
        immune_counts.set_data([], [])
        death_counts.set_data([], [])
        hospital.set_data([], [])
        rect.set_edgecolor('none')
        return particles, particles_sick, healthy_counts, sick_counts, immune_counts, death_counts, hospital, rect

    def animation(self):
        """perform animation step"""
        global box, ax, fig, dt, rect, particles, particles_sick, healthy_counts, sick_counts, immune_counts, \
            death_counts, hospital, particles_immune, particles_death, sick_count, death_count

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

        hospital.set_data([0, 200], [30, 30])

        healthy_counts.set_data(box.total_time, box.healthy_count)
        sick_counts.set_data(box.total_time, box.sick_count)
        immune_counts.set_data(box.total_time, box.immune_count)
        sick_count.set_xlim(0, max(box.total_time) + 3)
        sick_count.set_title(f"Healthy: {box.healthy_count[-1]}  Sick: {box.sick_count[-1]}  Recovered: {box.immune_count[-1]} Dead: {box.death_count[-1]}", fontsize=9)

        death_counts.set_data(box.total_time, box.death_count)
        death_count.set_xlim(0, max(box.total_time) + 3)

        return particles, particles_sick, sick_counts, particles_immune, \
               immune_counts, particles_death, death_counts, healthy_counts, hospital, rect

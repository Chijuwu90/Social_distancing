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
import random


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(self, init_state=None, bounds=None, size=0.01, m=0.05, quarantine_percentage=0):
        if bounds is None:
            bounds = [-2, 2, -2, 2]
        if init_state is None:
            init_state = [[1, 0, 0, -1],
                          [-0.5, 0.5, 0.5, 0.5],
                          [-0.5, -0.5, -0.5, 0.5]]
        self.init_state = np.asarray(init_state, dtype=float)
        self.m = m * np.ones(self.init_state.shape[0])
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
        self.immune_count = [0]
        self.death_list = []
        self.death_count = [0]
        self.total_time = [0.0]
        self.time_elapsed_factor = 1.2
        if quarantine_percentage == 0:
            self.quarantine_list = []
        else:
            self.quarantine_list = random.sample(range(1, 100), quarantine_percentage)

    def step(self, dt):

        # if abs(max(self.total_time) - 100) < 0.1:
        #    time.sleep(10)
        # else:
        #    pass

        """step once by dt seconds"""
        self.time_elapsed = self.time_elapsed + dt

        moving_dots = self.get_movable_dots()

        # update positions
        self.state[moving_dots, :2] = self.state[moving_dots, :2] + dt * self.state[moving_dots, 2:]

        # find collision pairs and update sick list
        ind1, ind2 = self.find_pairs_of_collisions()

        # Update sick count and total time
        self.update_counting_list()

        # update velocities of colliding pairs
        self.update_collision_velocity(ind1, ind2)

        # update immune
        immune_threshold_time = 100 * dt
        self.update_immune_list(immune_threshold_time)

        # update death
        self.update_death_list()

    def update_death_list(self):
        death = self.random_death()
        if death is False:
            pass
        else:
            death_ind = random.randint(0, len(self.sick_list) - 1)
            self.death_list = self.death_list + [self.sick_list[death_ind]]
            del self.sick_list[death_ind]
            del self.sick_last[death_ind]
            del self.sick_timing[death_ind]

    def update_immune_list(self, immune_threshold_time):
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

    def update_collision_velocity(self, ind1, ind2):
        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.m[i1]
            m2 = self.m[i2]

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

    def update_counting_list(self):
        self.sick_last = [self.time_elapsed - self.sick_timing[i] for i, _ in enumerate(self.sick_timing)]
        self.healthy_list = set(np.arange(100)) - set(self.sick_list) - set(self.immune_list) - set(self.death_list)
        self.healthy_count.append(len(self.healthy_list))
        self.sick_count.append(len(self.sick_list))
        self.immune_count.append(len(self.immune_list))
        self.death_count.append(len(self.death_list))
        self.total_time.append(self.time_elapsed * self.time_elapsed_factor)

    def get_movable_dots(self):
        not_moving_dots = self.death_list + self.quarantine_list
        moving_dots = list(set(np.arange(100)) - set(not_moving_dots))
        return moving_dots

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

    def find_pairs_of_collisions(self):
        # find pairs of particles undergoing a collision
        distance = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(distance < 7 * self.size)

        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # Update collision list
        contact_pairs = list(zip(ind1, ind2))
        for i in contact_pairs:
            if len(set(self.sick_list) & set(list(i))) > 0:
                new_sick = list((set(self.sick_list) | set(i)) - (set(self.sick_list) & set(i)))
                for new in new_sick:
                    if (new not in self.sick_list) & (new not in self.immune_list) & (new not in self.death_list):
                        # update sick list
                        self.sick_list = self.sick_list + [new]
                        # update the timing
                        self.sick_timing = self.sick_timing + [self.time_elapsed]
                    else:
                        pass
        return ind1, ind2

"""
Created on 29.03.2024

@author: Mathias Berg Rosendal (I'm not an epidemiologist!! This is just made for fun)
Also developed using OpenAI. (2024). ChatGPT (3.5) [Large language model]. https://chat.openai.com

Consist primarily of three classes: 
- Infection with parameters name, lifetime, radius, infect_rate and post_sick_resilience
- Person with parameters pos, resilience, direction, and more
- EpidemicModel with functions for infecting people, positioning them wrt. impassable objects and more

A file "Simulation_%parameters%.gif" is output from this script, which plots the movement and states of
people, impassable objects and the total number of healthy, infected and recovered people wrt time
"""

#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D
from PIL import Image

style = 'report'

if style == 'report':
    plt.style.use('default')
    fc = 'white'
elif style == 'ppt':
    plt.style.use('dark_background')
    fc = 'none'

#%% ------------------------------- ###
###     1. Classes and Functions    ###
### ------------------------------- ###

class Infection:
    def __init__(self, name: str, lifetime: int, radius: int,
                 infect_rate: float, post_sick_resilience: float) -> None:
        self.name = name
        self.lifetime = lifetime
        self.radius = radius
        self.infect_rate = infect_rate
        self.post_sick_resilience = post_sick_resilience

class Person:
    def __init__(self, pos: np.ndarray, delta_max: int,
                 infected: bool = False, recovered: bool = False,
                 resilience: float = 0.0, direction: str = None, 
                 **kwargs):
        
        self.pos = pos
        self.healthy = not infected
        self.infected = infected
        self.recovered = recovered
        self.resilience = resilience
        self.delta_max = delta_max
        
        self.set_direction(direction, **kwargs)
        
        
    def set_direction(self, direction: str = None,
                      low_prob: float = 0.03, high_prob: float = 1,
                      possible_to_stay: bool = False):
        
        self.prob = np.ones((self.delta_max*2 + 1, self.delta_max*2 + 1)) 
        if direction == 'random':
            direction = np.random.choice(['up', 'upleft', 'left', 
                                          'downleft', 'down', 'downright',
                                          'right', 'upright', None])
            
        if direction == 'up':
            self.prob[:, :] = low_prob
            self.prob[:delta_max+1, delta_max] = high_prob
        elif direction == 'upleft':
            self.prob[:, :] = low_prob
            self.prob[delta_max, delta_max] = high_prob
            for d in range(delta_max+1):
                self.prob[delta_max-d, delta_max-d] = high_prob
        elif direction == 'left':
            self.prob[:, :] = low_prob
            self.prob[delta_max, :delta_max+1] = high_prob
        elif direction == 'downleft':
            self.prob[:, :] = low_prob
            self.prob[delta_max, delta_max] = high_prob
            for d in range(delta_max+1):
                self.prob[delta_max+d, delta_max-d] = high_prob
        elif direction == 'down':
            self.prob[:, :] = low_prob
            self.prob[delta_max:, delta_max] = high_prob
        elif direction == 'downright':
            self.prob[:, :] = low_prob
            self.prob[delta_max, delta_max] = high_prob
            for d in range(delta_max+1):
                self.prob[delta_max+d, delta_max+d] = high_prob
        elif direction == 'right':
            self.prob[:, :] = low_prob
            self.prob[delta_max, delta_max:] = high_prob
        elif direction == 'upright':
            self.prob[:, :] = low_prob
            self.prob[delta_max, delta_max] = high_prob
            for d in range(delta_max+1):
                self.prob[delta_max-d, delta_max+d] = high_prob
            
        if not(possible_to_stay):
            self.prob[delta_max, delta_max] = 0
        

    def move(self, grid, grid_x, grid_y, change_direction_at_imp: bool = False):  
        
        # Find options
        grid = grid[self.slice_bounds()].copy()
        options = [(i,j) for i,j in zip(grid_x[self.slice_bounds()].flatten(),
                                        grid_y[self.slice_bounds()].flatten())]
        
        # Check if up against an impassable object
        if change_direction_at_imp:
            impsum = - grid[grid == -1].sum()
            if impsum / np.size(grid) > 0.3:
                self.set_direction('random')
        
        # Find impossible options (occupied or impassable)
        idx = grid == 0
        
        grid[idx] = 1.0 
        grid[~idx] = 0.0
        
        # Factor probabilities from person direction
        grid = self.prob * grid 
        
        # Normalise
        grid = grid / grid.sum()
        
        # Make a choice
        try:
            choice = np.random.choice(len(options), p=grid.flatten().tolist())
                  
            # Randomly move within the bounds
            self.pos = options[choice]
        except ValueError:
            # Nowhere to go
            pass
            
    def infect(self, time: int):
        self.infected = True
        self.recovered = False
        self.healthy = False
        self.infect_time = time

    def is_healthy(self):
        return self.healthy
    
    def is_infected(self):
        return self.infected
    
    def is_recovered(self):
        return self.recovered
    
    def slice_bounds(self):
        slice_x = slice(self.pos[0]-self.delta_max, self.pos[0]+self.delta_max+1)
        slice_y = slice(self.pos[1]-self.delta_max, self.pos[1]+self.delta_max+1)
        return slice_x, slice_y


class EpidemicModel:
    def __init__(self, num_people: int, area_size: tuple):
        self.num_people = num_people
        self.area_size = area_size
        
        # Generate grid and coordinates 
        self.grid_sta = np.zeros(area_size, dtype=int)
        self.grid_x, self.grid_y = np.indices((area_size[0], area_size[1]))

        # Insert impassable static points
        self.grid_sta[:,  :2] = -1
        self.grid_sta[:, -2:] = -1
        self.grid_sta[:2,  :] = -1
        self.grid_sta[-2:, :] = -1
        
        self.grid = self.grid_sta.copy()
    
    def insert_wall(self, wall_position: int,
                    direction: str = 'both', radius: int = 2, 
                    doors: bool = True, door_thick: int = 2):
    
        # Scale to area size
        door_thick = int(door_thick * area_size[1] / 70)
        
        # Insert impassable wall with door
        if (direction == 'both') | (direction == 'vertical'):
            mid1 = int(area_size[1]/4)
            
            # vertical wall
            self.grid_sta[wall_position-radius:wall_position+radius, :] = -1
            self.grid_sta[wall_position-radius:wall_position+radius, :] = -1

            # Two doors
            if doors:
                self.grid_sta[wall_position-radius:wall_position+radius,mid1-door_thick:mid1+door_thick] = 0
                mid1 += 2*mid1
                self.grid_sta[wall_position-radius:wall_position+radius,mid1-door_thick:mid1+door_thick] = 0
        
        if (direction == 'both') | (direction == 'horisontal'):
            # horisontal wall
            mid1 = int(area_size[0]/4)
            self.grid_sta[:, wall_position-radius:wall_position+radius] = -1
            self.grid_sta[:, wall_position-radius:wall_position+radius] = -1
            
            # Two doors
            if doors:
                self.grid_sta[mid1-door_thick:mid1+door_thick, wall_position-radius:wall_position+radius] = 0
                mid1 += 2*mid1
                self.grid_sta[mid1-door_thick:mid1+door_thick, wall_position-radius:wall_position+radius] = 0
            
        self.grid = self.grid_sta.copy()
        
    def insert_squares(self):        
        # Insert 7 random sized squares at hardcoded places
        N_squares = 7
        scale_factor_x = area_size[0] / 70
        scale_factor_y = area_size[1] / 50
        origins = [(10*scale_factor_x, 15*scale_factor_y), (40*scale_factor_x, 20*scale_factor_y), 
                    (33*scale_factor_x, 40*scale_factor_y), (58*scale_factor_x, 32*scale_factor_y), 
                    (60*scale_factor_x, 15*scale_factor_y),
                    (14*scale_factor_x, 35*scale_factor_y), (25*scale_factor_x, 10*scale_factor_y)]
        widths = [np.random.randint(4*scale_factor_x, 15*scale_factor_x) for i in range(N_squares)]
        lengths = [np.random.randint(4*scale_factor_y, 15*scale_factor_y) for i in range(N_squares)]
        for i,square in enumerate(origins):
            self.grid_sta[int(square[0]-widths[i]/2):int(square[0]+widths[i]/2),
                            int(square[1]-lengths[i]/2):int(square[1]+lengths[i]/2)] = -1
        
        self.grid = self.grid_sta.copy()
        
    def populate(self, init_infected: int,
                 delta_max: int, direction: str = 'random'):
        
        # Find passable set of coordinates
        idx = self.grid == 0
        passable = [(i,j) for i,j in zip(self.grid_x[idx],
                                          self.grid_y[idx])]
        
        # Populate with people
        self.people = []
        for _ in range(self.num_people):
            choice = np.random.randint(len(passable))
            self.people.append(Person((passable[choice][0],
                                      passable[choice][1]),
                                      delta_max,
                                      direction=direction,
                                      possible_to_stay=False))
            passable.pop(choice)
            
        for i, person in enumerate(self.people):
            self.grid[person.pos[0], person.pos[1]] = i + 1  # Update grid with person indices

        for _ in np.random.randint(0, num_people, init_infected):
            self.people[_].infect(0)  # Initially infect some people


    def move_people(self, change_direction_at_imp: bool = False):
        for i, person in enumerate(self.people):
            # Make staying on the spot an option
            x, y = person.pos
            self.grid[x, y] = 0
            
            # Move person
            person.move(self.grid,
                        self.grid_x,
                        self.grid_y,
                        change_direction_at_imp)
            
            # Update grid
            x, y = person.pos
            self.grid[x, y] = i + 1


    def infect_people(self, time: int, disease: Infection):
        for person in self.people:
            if not person.is_infected():
                continue
            if (time - person.infect_time) <= disease.lifetime:
                x, y = person.pos
                neighbors = self.grid[max(x - disease.radius, 0):min(x + disease.radius + 1, self.area_size[0]), 
                                      max(y - disease.radius, 0):min(y + disease.radius + 1, self.area_size[1])]
                unique_neighbors = np.unique(neighbors)
                unique_neighbors = unique_neighbors[unique_neighbors > 0]  # Filter people only
                for neighbor_idx in unique_neighbors:
                    neighbor = self.people[neighbor_idx - 1]
                    if not neighbor.is_infected() and np.random.rand() < disease.infect_rate - neighbor.resilience:
                        neighbor.infect(time)
            
            elif (time - person.infect_time) > disease.lifetime:
                person.infected = False
                person.recovered = True
                person.resilience += disease.post_sick_resilience

    def plot(self, time: int, data: list, max_iterations: int):
        fig, (ax, ax2) = plt.subplots(figsize=(8, 10), nrows=2, dpi=100,
                                      gridspec_kw={'height_ratios' : [4, 1],
                                                   'hspace' :0.01,})
        for person in self.people:
            if person.is_infected():
                p1, = ax.plot(person.pos[0], person.pos[1], 'o', color='#932E06')  # Infected person
            elif person.is_recovered():
                p3, = ax.plot(person.pos[0], person.pos[1], 'o', color='#86B5CD')  # Recovered person
            else:
                p2, = ax.plot(person.pos[0], person.pos[1], 'o', color='#02427F')  # Healthy person
        
        # Plot impassable objects
        idx = self.grid == -1
        impx, impy = self.grid_x[idx], self.grid_y[idx]
        scale_factor = np.sqrt(self.area_size[0]**2 + self.area_size[1]**2) / np.sqrt(70**2 + 50**2) 
        ax.plot(impx, impy, marker='s', linestyle='',
                markersize=6/scale_factor, color=[0,0,0,1])
        ax.set_aspect('equal')
        # ax.set_title('$t\t=\t%0.2f$'%time)
        ax.axis('off')
        ax.set_xlim(-.5, self.area_size[0]-.5)
        ax.set_ylim(-.5, self.area_size[1]-.5)
        if (not 'p1' in locals()) | (not 'p2' in locals()) | (not 'p3' in locals()):
            # Create a patch
            p1 = Line2D([0], [0], marker='o', linestyle='', color='#932E06')
            p2 = Line2D([0], [0], marker='o', linestyle='', color='#02427F')
            p3 = Line2D([0], [0], marker='o', linestyle='', color='#86B5CD')
            
        ax.legend((p2, p1, p3), ('Healthy', 'Infected', 'Recovered'),
            loc='center', bbox_to_anchor=(0.5, 1.05),
            ncol=3)
        
        # Plot infected vs healthy
        data[0].append(time) # time
        N_healthy = sum([person.is_healthy() for person in self.people])
        N_infected = sum([person.is_infected() for person in self.people])
        N_recovered = sum([person.is_recovered() for person in self.people])
        data[1].append(len(self.people) - N_infected - N_recovered) # healthy
        data[2].append(N_infected) # infected
        data[3].append(N_recovered) # recovered
        if len(data[0]) > 0:
            p1 = ax2.fill_between(data[0], 0,       data[2], color='#932E06', edgecolor='none') # infected
            temp = np.array(data[2]) + np.array(data[1])
            p2 = ax2.fill_between(data[0], data[2], temp, color='#02427F', edgecolor='none') # healthy
            p3 = ax2.fill_between(data[0], temp, temp + np.array(data[3]), color='#86B5CD', edgecolor='none') # recovered
        ax2.set_xlim(0, max_iterations)
        ax2.legend((p2, p1, p3), ['Healthy', 'Infected', 'Recovered'],
                   loc='center', bbox_to_anchor=(0.5, 1.15),
                    ncol=3)
        ax2.set_ylabel('# of People')
        ax2.set_xlabel('Time')
        
        return fig, (ax, ax2), data

#%% ------------------------------- ###
###           2. Run Code           ###
### ------------------------------- ###

if __name__ == '__main__':
    # Parameters
    num_people = 400
    area_size = (150, 100)
    init_infected = 1
    lockdown_limit = 40
    delta_max = 2
    total_iterations = 350
    direction = 'random'
    change_direction_time = 20
    change_direction_at_imp = True
    delta_t = 50 # duration in ms for gif

    # Disease
    lifetime = 50
    infect_radius = 2
    infect_rate = 0.3
    post_sick_resilience = 0.15
    disease = Infection('H1N1', lifetime, infect_radius, 
                        infect_rate, post_sick_resilience)

    model = EpidemicModel(num_people, area_size)

    # Create environment
    model.insert_wall(direction='vertical', wall_position=int(area_size[0]/2), door_thick=5)
    # model.insert_wall(direction='horisontal', wall_position=int(area_size[1]/2), door_thick=5)
    model.populate(init_infected, delta_max, direction)
    locked_down = False

    # Simulate 
    images = []
    data = [[], [], [], []] # placeholder for infection data

    t0 = datetime.now()

    # Make figure
    fig, (ax, ax2), data = model.plot(0, data, total_iterations)
    fig.canvas.draw()  # Draw the figure
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), 
                                fig.canvas.tostring_rgb())
    images.append(image)
    plt.close(fig)
    print('Simulating infection of %s\n'%disease.name)
    print('Iter\tElapsed (s)\tHealthy\tInfected\tRecovered')
    for i in range(1, total_iterations+1):
        t1 = datetime.now()
        print('%d\t%0.2f\t\t%d\t%d\t\t%d'%(i, (t1-t0).total_seconds(), data[1][-1], data[2][-1], data[3][-1]))

        # Do simulation
        model.move_people(change_direction_at_imp)
        model.infect_people(i, disease)
        
        
        # if locked_down and data[2][-1] == lockdown_limit/4
        
        # Change direction for every 100th iteration
        # if (i % change_direction_time == 0) & (i != 0):
        #     for person in model.people:
        #         person.set_direction('random')
        
        # Make figure
        fig, (ax, ax2), data = model.plot(i, data, total_iterations)
        fig.canvas.draw()  # Draw the figure
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), 
                                fig.canvas.tostring_rgb())
        images.append(image)
        plt.close(fig)
        
        # Lock doors if more than lockdown_limit infected
        if data[2][-1] > lockdown_limit:
            model.insert_wall(direction='vertical', wall_position=int(area_size[0]/2), door_thick=1)
            # model.insert_wall(direction='horisontal', wall_position=int(area_size[1]/2), door_thick=1)
            model.insert_wall(direction='vertical', wall_position=int(area_size[0]/4), door_thick=1)
            # model.insert_wall(direction='horisontal', wall_position=int(area_size[1]/4), door_thick=1)
            model.insert_wall(direction='vertical', wall_position=int(area_size[0]/4*3), door_thick=1)
            # model.insert_wall(direction='horisontal', wall_position=int(area_size[1]/4*3), door_thick=1)
            locked_down = True
        
        # print('Figure done')
    

    # Save the GIF
    if len(images) > 1:
        print('\nMaking gif...')
        images[0].save('Simulation_IR%0.3f_PSR%0.3f_T%d_LDL%d.gif'%(disease.infect_rate,
                                               disease.post_sick_resilience,
                                               disease.lifetime, lockdown_limit), save_all=True, append_images=images[1:], 
                    duration=delta_t, loop=0)
        print('Done')
        
    t2 = datetime.now()

    print('\nTotal time elapsed: %0.2f s\n'%(t2-t0).total_seconds())
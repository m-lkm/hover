from scipy.integrate import solve_ivp
import numpy as np
from Agent import Actor, Critic
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer 
from  matplotlib import pyplot as plt
import torch
from collections import deque
from heapq import heappush, heappop
from copy import deepcopy
import pickle


# TODO

# committe member score out of date
# add an Evaluate

# schedule learning rate decay

# consider seperating actor and critic learning rates

# loose grid search for meta parameters

# give each branch a time limit

# kinda screwy to use dist / value in training then select on survival

# Save/Load committee

# turn all the committee params into normal paramters 

# can add a critic trailer for bellman targeting

# add a cost for omega

# either tensorboard or plt logging
# maybe log or live plot actor and critic loss
# value of initial state

# dont let props thrust down?

# add a print

# add a Save / Load
# periodic saving

# see mode collapse

# see expanding circle as it gets better
# not sure about the starting at rest thing



# NOTES

# prioritized buffer not working well
# and slower
# not worth it
# maybe most of a batch from regular, bit from prioritized


# committees raise questions about independence

# state = (x, y, theta, x', y', omega)

# normalizing on reward instead of value

# trying uzbek process 


class Copter:
    def __init__(self, mass=1.0, arm_length=0.5, max_lift=10.0, time_step=1.0/30, max_dist=1.0, max_omega=5.0, max_num_steps=500, max_initial_angle_mag=np.pi/12, actor_num_hidden=50, critic_num_hidden=50, gamma=0.999, replay_buffer_capacity=10**5, lr=1e-3):
        self.mass = mass
        self.arm_length = arm_length
        self.max_lift = max_lift
        self.time_step = time_step
        self.g = 9.81

        self._lr = lr

        self.committee_size=11
        self.committee_heap = [] # (score, actor)
        self.min_turns_betweeen_committee_adds = 2000
        self.num_turns_til_you_can_add = self.min_turns_betweeen_committee_adds
        self.window_size = 100
        self.dq = deque()
        self.window_sum = 0

        

        self._gamma = gamma # property
        self._max_dist = max_dist # property

        self.r_lose = 1.0 / (self.gamma - 1) # v_worst 

        self.max_initial_dist_from_target = 0.5 * self.max_dist
        
        self.max_omega_mag = max_omega
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        # self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_capacity)

        # self.reward_scaling_factor = (gamma - 1.0) / max_dist
        
        self.actor = Actor(num_hidden=actor_num_hidden,
                           max_output=max_lift,
                           lr=lr)

        self.critic = Critic(num_hidden=critic_num_hidden,
                             gamma=gamma,
                             lr=lr,
                             output_scalar=self.r_lose)
        
        self.state = np.array([0] * 6)
        self.prop_force_1 = 0
        self.prop_force_2 = 0 
        self.max_num_steps = max_num_steps
        self.step = 0
        self.max_initial_angle_mag = max_initial_angle_mag
        # r_worst = -self.max_dist
        # v_worst = r_worst / (1.0 - gamma)
        # self.r_loss = 2 * v_worst - 1 # its worse to lose than to do bad
        plt.ion()
        return

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, val):
        self._lr = val
        for group in self.actor.optimizer.param_groups:
            group["lr"] = val
        for group in self.critic.optimizer.param_groups:
            group["lr"] = val
        return 
        
    @property
    def max_dist(self):
        return self._max_dist

    @max_dist.setter
    def max_dist(self, value):
        self._max_dist = value 
        self.max_initial_dist_from_target = 0.5 * value
        # self.reward_scaling_factor = (self.gamma - 1.0) / value
        return

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.critic.gamma = value
        # self.reward_scaling_factor = (self.gamma - 1.0) / value
        sel.r_lose = 1 / (self.gamma - 1)
        self.critic.output_scalar = self.r_lose
        return 

    def FormatPlot(self):
        plt.title(f"step = {self.step}\n$\omega = $%.1f"%self.state[5])
        plt.xlim(-self.max_dist - self.arm_length, self.max_dist + self.arm_length)
        plt.ylim(-self.max_dist - self.arm_length, self.max_dist + self.arm_length)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect("equal")
        self.DrawCircle(radius=self.max_dist, alpha=0.3)
        plt.legend()
        ax = plt.gca()
        plt.axhline(color="k", alpha=0.3)
        plt.axvline(color="k", alpha=0.3)
        return

    def DrawCircle(self, radius=1, num_points=200, alpha=1.0):
        theta = np.linspace(0, 2*np.pi, num_points + 1)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        plt.plot(x,y,alpha=alpha, c="blue", label= f"radius = {self.max_dist}")
        return 

    def Plot(self):
        plt.pause(0.1)
        plt.clf()
        self.FormatPlot()
        
        prop_length = self.arm_length/ 2
        stem_length = self.arm_length/4

        x_0, y_0, theta = self.state[:3]
        base_point = np.array([x_0, y_0])
        right_base_point = base_point + self.arm_length * np.array([np.cos(theta), np.sin(theta)])
        left_base_point = base_point - self.arm_length * np.array([np.cos(theta), np.sin(theta)])

        stem_vec = stem_length * np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
        right_prop_mid_point = right_base_point + stem_vec
        left_prop_mid_point = left_base_point + stem_vec
        
        prop_vec = prop_length * np.array([np.cos(theta), np.sin(theta)])
        right_prop_right_endpoint = right_prop_mid_point + prop_vec
        right_prop_left_endpoint = right_prop_mid_point - prop_vec
        left_prop_right_endpoint = left_prop_mid_point + prop_vec
        left_prop_left_endpoint = left_prop_mid_point - prop_vec

        plt.scatter(*base_point, c="k")
        plt.plot(*zip(left_base_point, right_base_point), c="k")
        plt.plot(*zip(left_base_point, left_prop_mid_point), c="k")
        plt.plot(*zip(right_base_point, right_prop_mid_point), c="k")
        plt.plot(*zip(left_prop_left_endpoint, left_prop_right_endpoint), c="r")
        plt.plot(*zip(right_prop_left_endpoint, right_prop_right_endpoint), c="g")
        return 
    
    def Train(self, batch_size=2, display=False):
        self.Reset()
        game_over = False
        while not game_over and self.step < self.max_num_steps:        
            a = self.actor(self.state, training=True).flatten().detach().numpy() # see 
            self.prop_force_1, self.prop_force_2 = a
        
            # here
            # maybe pull out into GetNextState
            s_prime = solve_ivp(copter.GetStatePrime, [0,self.time_step], copter.state,  t_eval=[self.time_step], method="RK23").y.flatten()

            game_over = self.IsGameOver(s_prime)
            r = self.GetReward(s_prime)            
            
            if type(self.replay_buffer) == PrioritizedReplayBuffer:
                priority = self.critic.Train(self.state, a, r, s_prime, game_over, self.actor)[0]
                self.replay_buffer.AddExperience(priority, self.state, a, r, s_prime, game_over)
            else:
                self.replay_buffer.AddExperience(self.state, a, r, s_prime, game_over)

            '''
            s_mirror = self.MirrorState(self.state)
            s_mirror_prime = self.MirrorState(s_prime)
            a_mirror = self.MirrorAction(a)
            
            # print(f"{type(s_mirror) = }")
            # print(f"{type(a_mirror) = }")
            # print(f"{type(r) = }")
            # print(f"{type(s_mirror_prime) = }")
            # print(f"{type(game_over) = }")
            
            if type(self.replay_buffer) == PrioritizedReplayBuffer:
                mirror_priority = self.critic.Train(s_mirror, a_mirror, r, s_mirror_prime, game_over, self.actor)[0]
                self.replay_buffer.AddExperience(mirror_priority, s_mirror, a_mirror, r, s_mirror_prime, game_over)                
            else:
                self.replay_buffer.AddExperience(s_mirror, a_mirror, r, s_mirror_prime, game_over) 
            '''
            self.state = s_prime 
            self.step += 1
            if display:
                self.Plot()

            s_, a_, r_, s_prime_, game_over_ = self.replay_buffer.GetBatch(batch_size)
            actor_loss = self.actor.Train(s_, self.critic)
            critic_losses = self.critic.Train(s_, a_, r_, s_prime_, game_over_, self.actor)

            if type(self.replay_buffer) == PrioritizedReplayBuffer:
                self.replay_buffer.ReportPriorities(critic_losses)
        
        return self.step



    def Play(self, use_committee=False):
        self.Reset()
        game_over = False
        while not game_over and self.step < self.max_num_steps:
            if use_committee:
                a_arr = [actor(self.state) for _, actor in self.committee_heap]
                a  = torch.stack(a_arr).mean(0).flatten().detach().numpy()
            else:
                a = self.actor(self.state).flatten().detach().numpy() # see 
            self.prop_force_1, self.prop_force_2 = a
            s_prime = solve_ivp(copter.GetStatePrime, [0,self.time_step], copter.state,  t_eval=[self.time_step], method="RK23").y.flatten()
            game_over = self.IsGameOver(s_prime)
            r = self.GetReward(s_prime)
            self.state = s_prime 
            self.step += 1
            self.Plot()
        return self.step
    
    def IsGameOver(self, state):
        r = np.linalg.norm(state[:2])
        return r >  self.max_dist or abs(state[5]) > self.max_omega_mag

    def GetReward(self, state):
        if self.IsGameOver(state):
            return self.r_lose
        r = np.linalg.norm(state[:2])
        return -r / self.max_dist  if r > self.max_dist/2 else -r / (2 * self.max_dist)
    
    def GetRandomInitialState(self):
        r = np.random.uniform(0, self.max_initial_dist_from_target)
        theta_polar = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta_polar)
        y = r * np.sin(theta_polar)
        theta = np.random.uniform(-self.max_initial_angle_mag, self.max_initial_angle_mag)
        x_prime = 0
        y_prime = 0
        omega = 0
        return np.array([x,y,theta, x_prime, y_prime, omega])        

    def Reset(self):
        self.state = self.GetRandomInitialState()
        self.step = 0
        self.prop_force_1 = 0
        self.prop_force_2 = 0 
        return self.state

    def GetStatePrime(self, time, state):
        # time used by solver
        return np.array([
            state[3],
            state[4],
            state[5],
            -np.sin(state[2]) * (self.prop_force_1 + self.prop_force_2) / self.mass,
            np.cos(state[2]) * (self.prop_force_1 + self.prop_force_2) / self.mass  - self.g,
            (self.prop_force_1 - self.prop_force_2) * self.arm_length / self.mass,
            ])
    '''
    def MirrorState(self, state):
        # x -> -x
        # theta  -> -theta
        # v_x -> -v_x
        # omega -> -omega
        new_state = state.copy()
        new_state[0] *= -1
        new_state[2] *= -1
        new_state[2] %= 2*np.pi
        new_state[3] *= -1
        new_state[5] *= -1
        return new_state

    def MirrorAction(self, action):
        # return action.flip(0) # torch
        return np.array(list(action)[::-1])
    '''
    

if __name__ == "__main__":
    batch_size=2
    lr=1e-3
    max_dist=1
    actor_num_hidden = 400
    critic_num_hidden = 400
    max_num_steps = 1000
    copter = Copter(max_dist=max_dist,
                    actor_num_hidden=actor_num_hidden,
                    critic_num_hidden=critic_num_hidden,
                    lr=lr,
                    max_num_steps=max_num_steps)

    print(copter.actor)
    print(copter.critic)
    print(copter.replay_buffer)

    try:
        print("trying to load committee...")
        with open("./committee.pickle", "rb") as f:
            heap = pickle.load(f)
            copter.committee_heap = heap
        print("loaded committee")
    except:
        print("could not load committee")

    i_game = 0
    update_freq = 500
    report_freq = 100
    preformance_threshold = 1.5 * copter.committee_heap[0][0] if copter.committee_heap else float("-inf")
    max_preformance = 0
    best_actor = None 
    
    while True:
        report = i_game%report_freq==0
        survived = copter.Train(batch_size=batch_size)
        # copter.actor.DecayEpsilon()
        i_game += 1
        if i_game%update_freq==0:
            print("updating trailer...")
            copter.actor.UpdateTrailer()

        if report:
            print(f"survived {survived} steps")
            # print("copter.actor.prob_random = %.2e"%copter.actor.prob_random)
            # print()
        
        copter.dq.append(survived)
        copter.window_sum += survived
        if len(copter.dq) > copter.window_size:
            copter.window_sum  -= copter.dq.popleft()
        if copter.window_sum > max_preformance:
            max_preformance = copter.window_sum
            best_actor = deepcopy(copter.actor)
            print(f"new best = {max_preformance}")
        if copter.num_turns_til_you_can_add:
            copter.num_turns_til_you_can_add -= 1
        '''
        elif len(copter.committee_heap) < copter.committee_size or max_preformance > preformance_threshold:
            print("added committee member with score ", copter.window_sum)
            print(f"{len(copter.committee_heap)} / {copter.committee_size}")
            heappush(copter.committee_heap, (max_preformance, best_actor))
            best_actor = None
            max_preformance = 0 
            copter.dq.clear()
            copter.window_sum = 0
            copter.num_turns_til_you_can_add = copter.min_turns_betweeen_committee_adds
            if len(copter.committee_heap) > copter.committee_size:
                _ = heappop(copter.committee_heap)
                preformance_threshold = 1.5 * copter.committee_heap[0][0]
        '''

    '''
    heappush(copter.committee_heap, (max_preformance, best_actor))
    _ = heappop(copter.committee_heap)
    '''

    for i_game in range(20):
        copter.Play(True)
        # input()
        plt.pause(1)

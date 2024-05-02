import torch 
import torch.nn as nn
from copy import deepcopy
from math import log, exp 

# TODO

# pass uzbek params into actor.__init__


# NOTES

# max_rel_noise=0.03


class Actor(nn.Module):
    def __init__(self, num_hidden=50, max_output=1.0, lr=1e-5, prob_random=1.0, min_prob_random=0.05, prob_random_half_life = 5_000):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_inputs = 6
        self.num_outputs = 2
        self.max_output = max_output
        
        # here 
        max_rel_noise = 0.02
        self.max_noise = self.max_output * max_rel_noise
        self.max_noise_change = self.max_noise / 5
        self.noise = 0
        self.noise_decay_rate = 0.9 # magic, tune        

        '''
        self.min_prob_random = min_prob_random
        self.prob_random = max(prob_random, min_prob_random,0)
        self.prob_random = min(1, self.prob_random)
        self.prob_random_decay_rate = exp(log(0.5) / prob_random_half_life)
        '''
        
        # self.FC1 = nn.Linear(self.num_inputs, num_hidden - self.num_inputs)
        # self.FC2 = nn.Linear(num_hidden, num_hidden - self.num_inputs)
        self.FC1 = nn.Linear(self.num_inputs, num_hidden)
        self.FC2 = nn.Linear(num_hidden, num_hidden)
        self.FCOut = nn.Linear(num_hidden, self.num_outputs)
        # self.Activation = nn.Tanh()
        self.Activation = torch.cos
        self.max_output = max_output
        self.unit_uniform_distro = torch.distributions.uniform.Uniform(0, 1)
        self.noise_distro = torch.distributions.uniform.Uniform(-self.max_noise_change, self.max_noise_change)
        # self.noise_distro = torch.distributions.uniform.Uniform(-max_output, max_output)
        # self.noise_distro = torch.distributions.uniform.Uniform(-max_rel_noise * max_output, max_rel_noise * max_output)
        self.trailer = deepcopy(self)
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5, lr=lr) # weight decay as L2 regularization term        
        return

    '''
    def GetRandomMove(self, batch_size):
        return self.noise_distro.sample((batch_size, self.num_outputs))
    '''
    
    def GetGreedyMove(self, x):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).reshape(-1, self.num_inputs)
        y = self.Activation(self.FC1(x))
        # y = torch.hstack([x, y])
        y = self.Activation(self.FC2(y))
        # y = torch.hstack([x, y])
        y = self.Activation(self.FCOut(y))
        y = y * self.max_output
        return y

    def GetNoisyMove(self, x):
        x = self.GetGreedyMove(x)
        with torch.no_grad():
            noise = self.noise_distro.sample((self.num_outputs,))
            # print(noise)
            self.noise  = self.noise +  noise 
            self.noise = self.noise.clip(-self.max_noise, self.max_noise)
            self.noise = self.noise * self.noise_decay_rate

        x = x + self.noise
        x = x.clip(-self.max_output, self.max_output)
        return x        

    '''
    def GetEpsilonGreedyMove(self, x, epsilon):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).reshape(-1, self.num_inputs)
        batch_size, _ = x.shape
        use_random = self.unit_uniform_distro.sample((batch_size,1)) < epsilon
        return use_random * self.GetRandomMove(batch_size) + (~use_random) * self.GetGreedyMove(x)
    '''

    def forward(self, x, training=False):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).reshape(-1, self.num_inputs).to(self.device)

        # mirror_state_mask = mask
        # mirrored_state = self.F(x)
        # x = mask fuckery
        #

        mirror_state_mask = (x[:,0] < 0).reshape(-1,1)
        mirrored_states = self.MirrorState(x)
        x = x * (~mirror_state_mask) + mirrored_states * mirror_state_mask
        
        if training:
            x = self.GetNoisyMove(x)
        else:
            x = self.GetGreedyMove(x)

        mirrored_actions = self.MirrorAction(x)
        x = x * (~mirror_state_mask) + mirrored_actions * mirror_state_mask
        return x       

    def MirrorState(self, state):
        # x -> -x
        # theta  -> -theta
        # v_x -> -v_x
        # omega -> -omega
        # new_state = state.copy()
        state[:,0] *= -1
        state[:,2] *= -1
        state[:,2] %= (2*torch.pi)
        state[:,3] *= -1
        state[:,5] *= -1
        return state

    def MirrorAction(self, action):
        # torch has problem with negative slicing
        return torch.vstack([action[:,1], action[:,0]]).transpose(0,1)

    def DecayEpsilon(self):
        self.prob_random *= self.prob_random_decay_rate
        self.prob_random = max(self.prob_random, self.min_prob_random)
        return 
    
    def UpdateTrailer(self, frac_new=1.0):
        frac_old = 1.0 - frac_new
        for new, old in zip(self.parameters(), self.trailer.parameters()):
            old.data = frac_old * old.data + frac_new * new.data
        return

    def Train(self, s, critic):
        # print("in actor.Train")
        self.optimizer.zero_grad()
        a = self(s)
        q = critic(s, a)
        # print(f"{q = }")
        loss = -q.mean()
        loss.backward()
        self.optimizer.step()
        return float(loss)




class Critic(nn.Module):
    def __init__(self, num_hidden=50, gamma=0.99, lr=1e-5, output_scalar=1):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_inputs = 6
        self.num_outputs = 2
        self.gamma=gamma

        # print("in critic.__init__")
        # print(f"{num_hidden = }")

        # self.FC1 = nn.Linear(self.num_inputs + self.num_outputs, num_hidden - (self.num_inputs + self.num_outputs))
        # self.FC2 = nn.Linear(num_hidden, num_hidden - (self.num_inputs + self.num_outputs))
        self.FC1 = nn.Linear(self.num_inputs + self.num_outputs, num_hidden)
        self.FC2 = nn.Linear(num_hidden, num_hidden)
        self.FCOut = nn.Linear(num_hidden, 1)
        # self.Activation = torch.abs
        self.Activation = torch.cos
        self.OutputActivation = nn.Sigmoid()
        self.output_scalar = output_scalar
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5, lr=lr) # weight decay as L2 regularization term
        self.Loss = torch.nn.MSELoss()
        return

    def FormatBatch(self, s, a, r, s_prime, game_over):
        # print("in FormatBatch")
        # print(f"{type(s) = }")
        # print(f"{type(a) = }")
        # print(f"{type(r) = }")
        # print(f"{type(s_prime) = }")
        # print(f"{type(game_over) = }")
        
        if type(s) != torch.Tensor:
            s = torch.Tensor(s)

        if type(a) != torch.Tensor:
            a = torch.Tensor(a)

        try:
            int(r)
            r = torch.Tensor([r])
        except:
          if type(r) != torch.Tensor:
            r = torch.Tensor(r)
        
        if type(s_prime) != torch.Tensor:
            s_prime = torch.Tensor(s_prime)

        try:
            int(game_over)
            game_over = torch.Tensor([game_over])
        except:
          if type(game_over) != torch.Tensor:
            game_over = torch.Tensor(game_over)

        # print(*[x.shape for x in (s, a, r, s_prime, game_over)], sep="\n")
        return [x.reshape(-1, x.shape[-1]).to(self.device) for x in (s, a, r, s_prime, game_over)]

    def Train(self, state, action, reward, next_state, game_over, actor):
        self.optimizer.zero_grad()
        state, action, reward, next_state, game_over = self.FormatBatch(state, action, reward, next_state, game_over)
        
        q = self(state, action)
        with torch.no_grad():
            a_prime = actor.trailer.GetGreedyMove(next_state)
            q_prime = self(next_state, a_prime)
            bellman_target = reward + (1-game_over) * self.gamma * q_prime
        losses = (q - bellman_target)**2
        loss = losses.mean()

        # print("in critic.Train")
        # print(f"{state.shape = }")
        # print(f"{action.shape = }")
        # print(f"{reward.shape = }")
        # print(f"{next_state.shape = }")
        # print(f"{game_over.shape = }")

        loss.backward()
        self.optimizer.step()
        return losses.detach().numpy().flatten()

    def MirrorState(self, state):
        # x -> -x
        # theta  -> -theta
        # v_x -> -v_x
        # omega -> -omega
        state[:,0] *= -1
        state[:,2] *= -1
        state[:,2] %= (2*torch.pi)
        state[:,3] *= -1
        state[:,5] *= -1
        return state

    def MirrorAction(self, action):
        # torch has problem with negative slicing
        return torch.vstack([action[:,1], action[:,0]]).transpose(0,1)

    def forward(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).reshape(-1, self.num_inputs)

        if type(action) != torch.Tensor:
            action = torch.Tensor(action).reshape(-1, self.num_outputs)

        mirror_state_mask = (state[:,0] < 0).reshape(-1,1)
        mirrored_states = self.MirrorState(state)
        mirrored_actions = self.MirrorAction(action)
        state = state * (~mirror_state_mask) + mirrored_states * mirror_state_mask
        action = action * (~mirror_state_mask) + mirrored_actions * mirror_state_mask
        x = torch.hstack([state, action])
        
        y = self.Activation(self.FC1(x))
        # y = torch.hstack([x,y])
        y = self.Activation(self.FC2(y))
        # y = torch.hstack([x,y])
        y = self.OutputActivation(self.FCOut(y))
        y = y * self.output_scalar
        return y 


if __name__ == "__main__":
    actor = Actor()
    print(actor)
    state = (0,) * 6
    y = Actor(state, training=True)
    print(y)



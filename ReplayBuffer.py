from random import randint, random
import numpy as np
import torch 
from os import getcwd
import pickle 

# TODO



# NOTES

# numpy buffer

# replace lowest priority experience

# sample experiences in proportion to priority

# use loss as priority 

# leaf:
# [priority, experience]

# non-leaf:
# (subtree_sum, subtree_min, subtree_argmin)



# mutability problem running np buffers?




class ReplayBuffer:
    def __init__(self, capacity, save_path=""):
        if not save_path:
            save_path = getcwd() + "\\replay_buffer.pickle"

        self.save_path = save_path
        self.capacity = capacity
        self.i = 0

        if self.Load():
            return
        
        self.s = []
        self.a = []
        self.r = []
        self.s_prime = []
        self.game_over = []
        return

    def __repr__(self):
        return "replay buffer %d / %d"%(len(self.s), self.capacity)

    def Save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)
        return

    def Load(self):
        print("trying to load replay buffer from memory...")
        buffer = None 
        try:
            with open(self.save_path, "rb") as f:
                buffer = pickle.load(f)
        except:
            print("could not load replay buffer from memory")
            return False

        self.s = buffer.s[:self.capacity]
        self.a = buffer.a[:self.capacity]
        self.r = buffer.r[:self.capacity]
        self.s_prime = buffer.s_prime[:self.capacity]
        self.game_over = buffer.game_over[:self.capacity]
        del buffer
        print("loaded replay buffer from memory")
        return True  

    def FormatExperience(self, s,a,r,s_prime,game_over):
        s = s.flatten()
        a = a.flatten()
        r = np.array([r])
        s_prime = s_prime.flatten()
        game_over = np.array([float(game_over)])

        if type(s) == torch.Tensor:
            s = s.detach().numpy()

        if type(a) == torch.Tensor:
            a = a.detach().numpy()

        if type(s_prime) == torch.Tensor:
            s_prime = s_prime.detach().numpy()

        return s,a,r,s_prime,game_over
    
    def AddExperience(self, s,a,r,s_prime,game_over):
        s, a, r, s_prime, game_over = self.FormatExperience(s,a,r,s_prime, game_over)
        if len(self.s) < self.capacity:
            self.s.append(s)
            self.a.append(a)
            self.r.append(r)
            self.s_prime.append(s_prime)
            self.game_over.append(game_over)
        else:
            self.s[self.i] = s
            self.a[self.i] = a
            self.r[self.i] = r
            self.s_prime[self.i] = s_prime
            self.game_over[self.i] = game_over
            self.i += 1
            self.i %= self.capacity
        return

    def GetBatch(self, batch_size):
        length = len(self.s)
        indices = [randint(0, length-1) for i in range(batch_size)]
        s = np.vstack([self.s[i] for i in indices])
        a = np.vstack([self.a[i] for i in indices])
        r = np.vstack([self.r[i] for i in indices])
        s_prime = np.vstack([self.s_prime[i] for i in indices])
        game_over = np.vstack([self.game_over[i] for i in indices])
        return s,a,r,s_prime,game_over



















class PrioritizedReplayBuffer:
    def __init__(self, capacity, save_path=""):
        if not save_path:
            save_path = getcwd() + "\\prioritized_replay_buffer.pickle"

        self.save_path = save_path
        self.capacity = capacity
        self.indices_waiting_for_priority_update = []
        self.experiences = [None] # offset for 1 indexing
        # (priority, experience)

        if self.Load():
            return 
        return

    def __repr__(self):
        return "prioritized replay buffer %d / %d"%(self.GetNumExperiences(), self.capacity)
    
    def Save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)
        return

    def Load(self):
        print("trying to load prioritized replay buffer from memory...")
        buffer = None 
        try:
            with open(self.save_path, "rb") as f:
                buffer = pickle.load(f)
        except:
            print("could not load prioritized replay buffer from memory")
            return False
        self.experiences = buffer.experiences
        # here
        # loaded too much problem
        # means we need to be able to pop leafs
        # bit of a pain...
        del buffer
        print("loaded prioritized replay buffer from memory")
        return True 

    def FormatExperience(self, s,a,r,s_prime,game_over):
        s = s.flatten()
        a = a.flatten()
        r = np.array([r])
        s_prime = s_prime.flatten()
        game_over = np.array([float(game_over)])

        if type(s) == torch.Tensor:
            s = s.detach().numpy()

        if type(a) == torch.Tensor:
            a = a.detach().numpy()

        if type(s_prime) == torch.Tensor:
            s_prime = s_prime.detach().numpy()

        return s,a,r,s_prime,game_over

    def GetNumExperiences(self):
        # start with length == 1
        # each time you add an experience
        return len(self.experiences)// 2

    def GetAvgPriority(self):
        num_experiences = self.GetNumExperiences()
        if not num_experiences:
            return 0
        total = self.experiences[1][0]
        return total / num_experiences    

    def AddExperience(self, priority, *experience):
        experience = self.FormatExperience(*experience)
        self.indices_waiting_for_priority_update.clear()
        num_experiences = self.GetNumExperiences()
        if not num_experiences:
            self.experiences.append([priority, experience])
            return

        if num_experiences == self.capacity:
            return self._ReplaceExperience(priority, experience)

        i_parent = num_experiences
        self.experiences.append(self.experiences[i_parent])
        self.experiences.append([priority, experience])
        self.Propagate(i_parent)
        return

    def FindLeftChild(self, i):
        return 2*i

    def FindRightChild(self, i):
        return 1 + self.FindLeftChild(i) 
        
    def Propagate(self, i):
        # print("in Propagate")
        # print(f"{i = }")
        
        while i:
            i_left = self.FindLeftChild(i)
            i_right = self.FindRightChild(i)
            if len(self.experiences[i_left]) == 2:
                left_sum = left_min  = self.experiences[i_left][0]
                left_argmin = i_left
            else:
                left_sum, left_min, left_argmin = self.experiences[i_left]

            if len(self.experiences[i_right]) == 2:
                right_sum = right_min  = self.experiences[i_right][0]
                right_argmin = i_right
            else:
                right_sum, right_min, right_argmin = self.experiences[i_right]
            subtree_sum = left_sum + right_sum
            subtree_min = min(left_min, right_min)
            subtree_argmin = left_argmin if left_min <= right_min else right_argmin
            self.experiences[i] = (subtree_sum, subtree_min, subtree_argmin)
            i>>=1
        return 

    def _ReplaceExperience(self, priority, experience):
        i_leaf = self.experiences[1][2]
        self.experiences[i_leaf] = [priority, experience]
        i_parent = i_leaf>>1
        if i_parent:
            self.Propagate(i_parent)
        return 

    def _SampleForExperience(self):

        # print()
        # print("in Sample")
        # print(f"{self.GetNumExperiences() = }")
        total = self.experiences[1][0]
        # print(f"{total = }")
        target = total * random()        
        i = 1
        while self.FindLeftChild(i) < len(self.experiences):
            # node if not a leaf
            i_left = self.FindLeftChild(i)
            i_right = self.FindRightChild(i)
            left_sum = self.experiences[i_left][0]
            right_sum = self.experiences[i_right][0]
            if left_sum >= target:
                i = i_left
            else:
                i = i_right
                target -= left_sum

        self.indices_waiting_for_priority_update.append(i)
        return self.experiences[i][1]

    def _UniformSampleForExperience(self):
        num_experiences = self.GetNumExperiences()
        i = len(self.experiences) - num_experiences
        j = len(self.experiences)
        i_exp = np.random.randint(i,j)
        self.indices_waiting_for_priority_update.append(i_exp)
        return self.experiences[i_exp][1]


    def GetBatch(self, batch_size):
        self.indices_waiting_for_priority_update.clear()
        # experiences  = [self._SampleForExperience() for i in range(batch_size)]
        experiences  = [self._UniformSampleForExperience() for i in range(batch_size)]
        return [np.vstack(arr) for arr in zip(*experiences)]

    def ReportPriorities(self, new_priorities):
        # print("in ReportExperiences")
        # print(f"{self.indices_waiting_for_priority_update = }")
        # print(f"{new_priorities = }")
        for i, priority  in zip(self.indices_waiting_for_priority_update, new_priorities):
            self.experiences[i][0] = priority
            i_parent = i>>1
            self.Propagate(i_parent)
        self.indices_waiting_for_priority_update.clear()
        return 

if __name__ == "__main__":
    capacity = 10
    replay_buffer = ReplayBuffer(capacity)

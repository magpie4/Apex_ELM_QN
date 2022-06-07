from oselm import OS_ELM
import numpy as np
import copy


class OS_ELM_DQN(object):
    def __init__(self, input_dim, hidden_dim, output_dim, gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.action_onehot = np.eye(output_dim)
        
        self.action_list = [i for i in range(output_dim)]
        
        self.q_network = OS_ELM(input_dim+output_dim, hidden_dim, 1)
        self.q_network.set_p()
        self.gamma = gamma
        self.target_q = copy.deepcopy(self.q_network)
        
    def __call__(self, states):
        q_list = []
        for state in states:
            tmp=[]
            for i in range(self.output_dim):
                state_action = np.concatenate([state,self.action_onehot[i]])
                tmp.append(self.q_network(state_action)[0][0])
            q_list.append(tmp)

        return q_list   

    def sample_action(self, state, epsilon):
        q_list = []
        
        if np.random.random() < epsilon:
            return np.random.choice(self.action_list)
        
        for i in range(self.output_dim):
            #print(state,self.action_onehot[i])
            state_action = np.concatenate([state,self.action_onehot[i]])
            q_list.append(self.q_network(state_action)[0][0])
        
        #print(q_list)
        return np.argmax(q_list)
    
    def reshape(self, state, action, reward, next_state, done):
        state = np.array(state).astype(np.float32).reshape(-1, self.input_dim)
        action = np.array(action).astype(np.float32).reshape(-1, self.output_dim)
        reward = np.array(reward).astype(np.float32).reshape(-1, 1)
        next_state = np.array(next_state).astype(np.float32).reshape(-1, self.input_dim)
        done = np.array(done).astype(np.float32).reshape(-1, 1)
        return state, action, reward, next_state, done
        
    def clipping(self, q_value):
        q_value[q_value > 1] = 1.0
        q_value[q_value < -1] = -1.0
        return q_value
    
    def compute_expected_value(self, r, next_s, d):
        """next_s_a = []

        for i in range(16):
            #print(next_s,np.broadcast_to(np.array(self.action_onehot[i]).reshape(-1,2),(len(next_s),self.output_dim)))
            action_bro = np.broadcast_to(np.array(self.action_onehot[i]).reshape(-1,2),(len(next_s),self.output_dim))
            print(action_bro,next_s)
            next_s_a.append(np.concatenate([next_s[0],action_bro[0]]))
        
        print(next_s_a)
        next_value=(np.concatenate([self.tartget_q(next_s_a)], axis=1))
        print(next_value)"""
        #print(next_s,np.broadcast_to(np.array(self.action_onehot[0]).reshape(-1,2),(len(next_s),self.output_dim)))
        next_s_a_0 = np.concatenate([next_s, np.broadcast_to(np.array(self.action_onehot[0]).reshape(-1,2),(len(next_s),self.output_dim))], axis=1)
        next_s_a_1 = np.concatenate([next_s, np.broadcast_to(np.array(self.action_onehot[1]).reshape(-1,2),(len(next_s),self.output_dim))], axis=1)
        next_value_0 = self.target_q(next_s_a_0)
        next_value_1 = self.target_q(next_s_a_1)
        #print(next_s_a_0)
        next_value_concat = np.concatenate([next_value_0, next_value_1], axis=1)
        next_value_max = next_value_concat.max(axis=1).reshape(-1, 1).astype(np.float32)
        expected_value = r + (1-d) * self.gamma * next_value_max
        #print(expected_value)
        expected_value = self.clipping(expected_value)
        return expected_value
        
        
        
    def init_learning(self, state, action, reward, next_state, done):
        action = np.identity(self.output_dim)[action]
        state, action, reward, next_state, done = self.reshape(state, action, reward, next_state, done)
        
        
        s_a = np.concatenate([state, action], axis=1)
        expected_value = self.compute_expected_value(reward, next_state, done)
        self.q_network.init_train(s_a, expected_value)
        
    def seq_learning(self, state, action, reward, next_state, done, weight=1):
        action = np.identity(self.output_dim)[action]
        state, action, reward, next_state, done = self.reshape(state, action, reward, next_state, done)

        s_a = np.concatenate([state, action], axis=1)
        expected_value = self.compute_expected_value(reward, next_state, done)
        self.q_network.seq_train(s_a, expected_value, weight)
    
    def get_weights(self):
        return [self.q_network.alpha,self.q_network.beta,self.q_network.bias,self.q_network.p]
    
    def set_weights(self, weight):
        self.q_network.alpha = weight[0]
        self.q_network.beta = weight[1]
        self.q_network.bias = weight[2]
        self.q_network.p = weight[3]      
        
    def soft_update(self, tau=0.95):
        self.target_q.alpha = tau * self.q_network.alpha + (1-tau) * self.target_q.alpha
        self.target_q.beta = tau * self.q_network.beta + (1-tau) * self.target_q.beta
        self.target_q.bias = tau * self.q_network.bias + (1-tau) * self.target_q.bias
        self.target_q.p = tau * self.q_network.p + (1-tau) * self.target_q.p
        
        
            
        

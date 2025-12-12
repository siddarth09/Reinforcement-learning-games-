import numpy 

grid_size=5 
gamma=0.99 #Discount factor
theta=1e-4 #Convergence threshold

#rewards 
goal = (4,4) 
step_reward=-3

#States
states = [(i,j) for i in range(grid_size) for j in range(grid_size)]

actions = {
    "U":(-1,0),
    "D":(1,0),
    "L":(0,-1),
    "R":(0,1)
}

arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

def next_state(state,action):
    if state == goal:
        return state 
    
    i,j = state 
    di,dj = actions[action]
    ni,nj = i+di ,j+dj 

    if ni<0 or ni>=grid_size or nj<0 or nj>= grid_size:
        return state 
    
    return (ni,nj)


#Value iteration
v= {s:0 for s in states} #initialize to zeroes 

while True:
    delta = 0 
    v_new= v.copy ()
    for s in states:
        if s == goal:
            continue 

        values = [] 
        for a in actions:
            s_next = next_state(s,a)
            reward= 0 if s_next == goal else step_reward 
            values.append(reward+gamma*v[s_next])

        v_new[s] = max(values)

        delta = max(delta,abs(v_new[s]-v[s]))

    v = v_new 
    if delta < theta: 
        break 

print("Value function:")
for i in range(grid_size):
    row = [f"{v[(i,j)]:6.2f}" for j in range(grid_size)]
    print(" ".join(row))


policy = {} 
for s in states:
    if s == goal:
        policy[s] = "G"
        continue 

    best_action = None 
    best_value  = -1e9

    for a in actions: 
        s_next = next_state(s,a)
        reward = 0 if s_next == goal else step_reward
        val = reward + gamma * v[s_next] 

        if val>best_value:
            best_value = val 
            best_action = a 

    policy[s]=arrow[best_action]

print("\nOptimal Policy:")
for i in range(grid_size):
    row = [policy[(i,j)] for j in range(grid_size)]
    print(" ".join(row))
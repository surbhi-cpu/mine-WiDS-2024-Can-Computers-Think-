import random
from more_itertools import distinct_permutations

class Puzzle:
    def __init__(self):
        self.gamma = 0.9
        self.num_cells = 16
      
        self.phase1_solved = [1, 2, 3, 4, self.num_cells] + [0]*11
        self.phase2_solved = [-1]*4 + [5, 6, 7, 8, self.num_cells] + [0]*7
        self.phase3_solved = [-1]*8 + [9, 10, 11, 12, 13, 14, 15, self.num_cells]

        
        self.phase1_states = self.generate_states([1, 2, 3, 4, 16], 0, 11)
        self.phase1_policy = self.policy_iteration(self.phase1_states)

        self.phase2_states = self.generate_states([5, 6, 7, 8, 16], 4, 7)
        self.phase2_policy = self.policy_iteration(self.phase2_states)

        self.phase3_states = self.generate_states([9, 10, 11, 12, 13, 14, 15, 16], 8, 0)
        self.phase3_policy = self.policy_iteration(self.phase3_states)

    def get_possible_actions(self, state):
       
        actions = []
        idx = state.index(self.num_cells)
        row, col = divmod(idx, 4)
        
        if row > 0: actions.append("Up")
        if row < 3: actions.append("Down")
        if col > 0: actions.append("Left")
        if col < 3: actions.append("Right")

        return actions

    def apply_move(self, state, action):
        
        idx = state.index(self.num_cells)
        row, col = divmod(idx, 4)

        if action == "Up" and row > 0 and state[idx - 4] != -1:
            state[idx], state[idx - 4] = state[idx - 4], state[idx]
        elif action == "Down" and row < 3 and state[idx + 4] != -1:
            state[idx], state[idx + 4] = state[idx + 4], state[idx]
        elif action == "Left" and col > 0 and state[idx - 1] != -1:
            state[idx], state[idx - 1] = state[idx - 1], state[idx]
        elif action == "Right" and col < 3 and state[idx + 1] != -1:
            state[idx], state[idx + 1] = state[idx + 1], state[idx]
        
        return state

    def is_partial_goal_state(self, state):
        
        return (state[:4] == self.phase1_solved[:4] or 
                state[:8] == self.phase2_solved[:8] or 
                state == self.phase3_solved)

    def get_reward(self, state):
        
        if self.is_partial_goal_state(state): return 1000
        else: return 0

    def mask_state(self, state, phase):
      
        if phase == 1:
            return [x if x in [1, 2, 3, 4, 16] else 0 for x in state]
        elif phase == 2:
            return [x if x in [5, 6, 7, 8, 16] else (-1 if x in [1, 2, 3, 4] else 0) for x in state]
        elif phase == 3:
            return [x if x in [9, 10, 11, 12, 13, 14, 15, 16] else -1 for x in state]
        return state

    def policy_improvement(self, states, policy, value_function):
      
        stable = True
        for state in states:
            state_tuple = tuple(state)
            prev_action = policy[state_tuple]
            
            if self.is_partial_goal_state(state):
                continue

            max_value = -float('inf')
            best_action = ''
            
            for action in self.get_possible_actions(state):
                next_state = self.apply_move(state.copy(), action)
                action_value = self.get_reward(next_state) + self.gamma * value_function.get(tuple(next_state), 0)
                
                if action_value > max_value:
                    max_value = action_value
                    best_action = action
            
            policy[state_tuple] = best_action
            if best_action != prev_action:
                stable = False
        
        return stable

    def policy_evaluation(self, states, policy, value_function):
       
        for state in states:
            state_tuple = tuple(state)
            if self.is_partial_goal_state(state):
                continue

            action = policy[state_tuple]
            next_state = self.apply_move(state.copy(), action)
            value_function[state_tuple] = self.get_reward(next_state) + self.gamma * value_function.get(tuple(next_state), 0)

    def policy_iteration(self, states):
        
        value_function = {tuple(state): 0 for state in states}
        policy = {tuple(state): '' for state in states}

        stable = False
        while not stable:
            stable = self.policy_improvement(states, policy, value_function)
            self.policy_evaluation(states, policy, value_function)
        
        return policy

    def generate_states(self, distinct_numbers, num_minus_ones, num_zeros):
        
        arr = distinct_numbers + [0] * num_zeros
        result = []
        for perm in distinct_permutations(arr):
            result.append([-1] * num_minus_ones + list(perm))
        return result

    def scramble_puzzle(self, num_steps=250):
      
        solved_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        current_state = solved_state.copy()
        
        for _ in range(num_steps):
            actions = self.get_possible_actions(current_state)
            action = random.choice(actions)
            current_state = self.apply_move(current_state, action)
        
        return current_state

    def display_state(self, state):
      
        for i in range(0, 16, 4):
            row = ["|"]
            for num in state[i:i + 4]:
                row.append(f" {num:2d} ")
            row.append("|")
            print("".join(row))
        print("----------------------------")

    def solve_puzzle(self, puzzle):
     
        state = puzzle.copy()

        def solve_phase(masked_state, policy):
            actions_taken = []
            while True:
                if tuple(masked_state) in policy:
                    action = policy[tuple(masked_state)]
                    if action not in ["Up", "Down", "Left", "Right"]:
                        break
                    actions_taken.append(action)
                    masked_state = self.apply_move(masked_state, action)
                else:
                    break
            return actions_taken

    # fOR 1 to 4
        phase1_masked_state = self.mask_state(state, 1)
        phase1_actions = solve_phase(phase1_masked_state, self.phase1_policy)

        for action in phase1_actions:
            state = self.apply_move(state, action)
            self.display_state(state)

   # FOR 5 to 8
        phase2_masked_state = self.mask_state(state, 2)
        phase2_actions = solve_phase(phase2_masked_state, self.phase2_policy)

        for action in phase2_actions:
            state = self.apply_move(state, action)
            self.display_state(state)

     #for 9 to 15
        phase3_masked_state = self.mask_state(state, 3)
        phase3_actions = solve_phase(phase3_masked_state, self.phase3_policy)

        for action in phase3_actions:
            state = self.apply_move(state, action)
            self.display_state(state)

        print("FINALLY hogaya")

# Initialize the puzzle game and solve it
puzzle_game = Puzzle()
scrambled_puzzle = puzzle_game.scramble_puzzle()
print("Lets go:")
puzzle_game.display_state(scrambled_puzzle)
puzzle_game.solve_puzzle(scrambled_puzzle)

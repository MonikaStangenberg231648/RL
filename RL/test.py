from sim_class import Simulation
import time

# Initialize the simulation with one robot
sim = Simulation(num_agents=1)

# Function to find the boundary for a given direction
def find_boundary(sim, direction, step_size=0.1, max_steps=100):
    position_history = []
    for _ in range(max_steps):
        # Set velocity in the specified direction
        velocity = [0.0, 0.0, 0.0]
        velocity[direction] = step_size
        actions = [[velocity[0], velocity[1], velocity[2], 0]]  # Drop command = 0
        
        # Run the simulation for one step
        state = sim.run(actions, num_steps=1)
        
        # Debug: Print the entire state to understand its structure
        print(f"State returned: {state}")
        
        # Check if 'position' key exists and extract position
        if 'robot_0' in state and 'position' in state['robot_0']:
            current_position = state['robot_0']['position']
            position_history.append(current_position)
        else:
            print("Error: 'robot_0' or 'position' key not found in state.")
            return None  # Stop if the structure is not as expected
        
        # Check if the robot stopped moving
        if len(position_history) > 1 and position_history[-1] == position_history[-2]:
            print(f"Boundary reached in direction {['X', 'Y', 'Z'][direction]}: {current_position}")
            return current_position
        
        time.sleep(0.05)  # Pause for visualization
    
    print(f"Maximum steps reached in direction {['X', 'Y', 'Z'][direction]}: {position_history[-1]}")
    return position_history[-1]

# Find boundaries in all 6 directions (X-, X+, Y-, Y+, Z-, Z+)
boundaries = {}
for direction in range(3):  # X, Y, Z
    boundaries[f"{['X', 'Y', 'Z'][direction]}-"] = find_boundary(sim, direction, step_size=-0.1)
    boundaries[f"{['X', 'Y', 'Z'][direction]}+"] = find_boundary(sim, direction, step_size=0.1)

# Print the working envelope
print("\nWorking Envelope Boundaries:")
for direction, position in boundaries.items():
    print(f"{direction}: {position}")


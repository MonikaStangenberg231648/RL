from sim_class import Simulation

sim = Simulation(num_agents=1)

velocity_x = -0.1
velocity_y = -0.1
velocity_z = -0.1
drop_command = 0  
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# List to track the robot's positions
tracked_positions = []

# Main simulation loop
for step in range(800):
    sim.run(actions, num_steps=1)  
    # Get the current position of the robot
    positions = sim.get_states()
    if positions:
        robot_position = positions.get("robotId_1", {}).get("robot_position")
        if robot_position:
            tracked_positions.append(robot_position)  # Add the robot's position to the list

if tracked_positions:
    x_positions = [pos[0] for pos in tracked_positions]
    y_positions = [pos[1] for pos in tracked_positions]
    z_positions = [pos[2] for pos in tracked_positions]

    min_x = min(x_positions)  # Maximum x position
    min_y = min(y_positions)  # Maximum y position
    min_z = min(z_positions)  # Maximum z position

    print(f"Min x: {min_x}, Min y: {min_y}, Min z: {min_z}")
else:
    print("No positions tracked.")




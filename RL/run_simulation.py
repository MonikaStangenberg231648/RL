from sim_class import Simulation

sim = Simulation(num_agents=1)  

velocity_x = 0.1
velocity_y = 0.1
velocity_z = 0.1
drop_command = 0  
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

for step in range(1000):
    sim.run(actions, num_steps=1)
    print(f"Step {step + 1}, Positions: {sim.droplet_positions}")
    
sim.run(actions, num_steps=1000)

positions = sim.droplet_positions
if positions:
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]
    z_positions = [pos[2] for pos in positions]

    max_x = max(x_positions)
    max_y = max(y_positions)
    max_z = max(z_positions)

    print(f"Max x: {max_x}, Max y: {max_y}, Max z: {max_z}")

print("Droplet Positions:", sim.droplet_positions)
print("States:", sim.get_states())
print("Pipette Positions:", sim.pipette_positions)
print("Robot IDs:", sim.robotIds)






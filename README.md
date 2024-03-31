# Simple SIR Semi-Guided Random Walk Epidemic Model

NOTE: I am no epidemiologist, this is just some ChatGPT lolz.

A very simple SIR (Susceptible, Infected, Recovered) model of disease spread.
Developed in cooperation with OpenAI. (2024). ChatGPT (3.5) [Large language model]. https://chat.openai.com

Consist primarily of three classes: 
- "Infection" with parameters name, lifetime, radius, infect_rate and post_sick_resilience
- "Person" with parameters pos, resilience, direction, and more
- "EpidemicModel" with f# Simple SIR Semi-Guided Random Walk Epidemic Model

A file "Simulation_%settings%.gif" is output from this script, which plots the movement and states of people, impassable objects, and the total number of healthy, infected and recovered people wrt. time. See example below:
![Simulation_IR0 300_PSR0 150_T50_LDL500](https://github.com/Mathias157/toy-epidemic-model/assets/77012503/0fea6da9-f9bc-45eb-9339-a77ff55d0f2f)

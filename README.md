##AXA - Driver Telematics Analysis

### Ideas

Features:
* triptime: total driving time
* standing time: standing time during a trip
* total distance
* skyway distance
* average velocity
* std (standard deviation) velocity
* average acceleration
* std acceleration
* average turing angle
* std of turning angle

For fast running. I try first with logistic regression. Score obtained 0.75

Features want to build:
* Percentiles of velocity, acceleration, turning angle
* turing aggression = angle of turn x velocity

### How to generate the solution
Just run "python main_lr.py"

### Settings in __main_
* num_cores: allow multiprocessing N jobs
* drivers: drivers to process and save to file


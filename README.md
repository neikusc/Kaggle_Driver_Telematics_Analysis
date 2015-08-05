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
* average of speeding up
* average of slowing down

Features want to build:
* Percentiles of velocity, acceleration, turning angle
* turing aggression = angle of turn x velocity


Models: 

* Predicting trips for each driver by adding random trips from other drivers.
For example, 200 trips of driver_001 are labeled as 1. numbNeg= 200 others drivers
are randomly chosen, one trip from these drivers is chosen, labelled as 0, and added to
above training set. This set then fitted by using LR model, fitting result then be 
applied back on 200 trips of driver_001. 
* Need run an experiment to search for an optimum  of numNeg
* For fast running. I try first with logistic regression. Score obtained 0.76089
*   

### How to generate the solution
* Just run "python main_lr.py"
* Deep Learning approach using Neon: I succeeded to run for single driver. But I could 
not solve a bug with gen_backend. File main_neon.py contains code finding probilities 
for single driver. For all drivers, code will be implemented as the same as file main_lr.py

### Settings in __main_
* num_cores: allow multiprocessing N jobs
* training set need to be created before input in any model by a function 
create_training_data()

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


Features want to build:
* Percentiles of velocity, acceleration, turning angle
* turing aggression = angle of turn x velocity


Models: 

* Predicting trips for each driver by adding random trips from other drivers.
For example, 200 trips of driver_001 are labeled as 1. numbNeg= 200 others drivers
are randomly chosen, one trip from these drivers is chosen, labelled as 0, and added to
above training set. This set then fitted by using LR model, fitting result then be 
applied back on 200 trips of driver_001. 

For fast running. I try first with logistic regression. Score obtained 0.75360

### How to generate the solution
Just run "python main_lr.py"

### Settings in __main_
* num_cores: allow multiprocessing N jobs
* drivers: drivers to process and save to file


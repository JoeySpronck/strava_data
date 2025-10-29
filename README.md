# STRAVA DATA: Weekly Running Plans & Progress
###### Joey Spronck
This repository tracks running progress and training plans. All data are updated via strava API.

## Checkout the [plot_updates README](https://github.com/JoeySpronck/strava_data/tree/plot_updates/README.md)
For automatically updated figures (via github actions).

---
### 📈 Weekly volume progression
###### Based on 10% rule. The next unreached target (this or next week) will be used as target for following plots
<p align="left">
  <img src="plots/weekly_distance_targets.png" alt="Weekly distance targets" width="500">
</p>

---
### Example Week Plans
###### These plots show examples of how a week could be divided into multiple runs, given the target mileage.
<p align="left">
  <img src="plots/week_plan_3_runs.png" alt="Week plan (3 runs)" width="300">
</p>
<p align="left">
  <img src="plots/week_plan_4_runs.png" alt="Week plan (4 runs)" width="300">
</p>
<p align="left">
  <img src="plots/week_plan_5_runs.png" alt="Week plan (5 runs)" width="300">
</p>

---
### Current/Next Week Plans
###### These plots automatically update and show the ran runs in white and proposed runs in orange. They try to somewhat stick to schemes plotted above.
<p align="left">
  <img src="plots/current_week_plan_3_runs.png" alt="Current week (3 runs)" width="300">
</p>
<p align="left">
  <img src="plots/current_week_plan_4_runs.png" alt="Current week (4 runs)" width="300">
</p>
<p align="left">
  <img src="plots/current_week_plan_5_runs.png" alt="Current week (5 runs)" width="300">
</p>

---
### Weekly Stacked Plots
###### Stacked barplots, showing run stacks for each week. 

#### Color = Risk 
###### Here risk is defined by combining distance from normal distribution. Faster and longer runs contribute to higher risk, slower and shorter to lower risk.
<p align="left">
  <img src="plots/weekly_risk.png" alt="Weekly risk" width="800">
</p>

#### Color = Pace 
<p align="left">
  <img src="plots/weekly_pace.png" alt="Weekly pace" width="800">
</p>

#### Color = Distance 
<p align="left">
  <img src="plots/weekly_distance.png" alt="Weekly distance" width="800">
</p>

---


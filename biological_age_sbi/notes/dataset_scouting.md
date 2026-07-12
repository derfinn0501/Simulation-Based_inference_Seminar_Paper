# Dataset Scouting: Simple Bioindicators

Goal: find datasets that can ground a first biological-age simulator with easy bioindicators such as eating behavior, sleep, heart-rate/fitness measures, and blood pressure.

## Best First Candidate: NHANES

NHANES is the easiest starting point.

Useful variables:

- age, sex, race/ethnicity
- dietary intake
- sleep duration and sleep-related questionnaire variables
- physical activity
- blood pressure
- height, weight, BMI, waist circumference
- laboratory markers such as cholesterol, glucose/HbA1c, vitamin D, etc.
- some cycles include cardiovascular fitness/treadmill data with heart-rate response and estimated VO2 max

Why it is useful:

- public and well documented
- cross-sectional population sample
- combines questionnaires, exams, and labs
- enough variables to build a simple conditional simulator

Main limitation:

- biological age is not directly observed
- simulator calibrated on chronological age may mostly learn chronological-age proxies
- cardiovascular fitness data are cycle-specific, not universal across all NHANES years

Sources:

- https://wwwn.cdc.gov/nchs/nhanes/
- https://odphp.health.gov/healthypeople/objectives-and-data/data-sources-and-methods/data-sources/national-health-and-nutrition-examination-survey-nhanes
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/CVX_C.htm

## Strong but Less Immediate: UK Biobank

Useful variables:

- age
- diet questionnaires
- sleep questionnaires
- blood pressure
- pulse/heart-rate measurements
- physical measurements
- physical activity and accelerometer data
- ECG/fitness-test fields, including maximum heart rate during fitness test

Why it is useful:

- very rich and large
- good for realistic correlation structure
- contains more sophisticated activity/fitness variables than NHANES

Main limitation:

- access controlled
- more setup overhead
- cohort is not a simple representative population sample

Sources:

- https://www.ukbiobank.ac.uk/about-our-data/types-of-data/physical-measurements/
- https://biobank.ctsu.ox.ac.uk/ukb/label.cgi?id=100012

## Sleep-Focused Candidate: MESA Sleep

Useful variables:

- age and demographic covariates
- sleep questionnaires
- actigraphy
- polysomnography
- selected MESA Sleep covariates
- broader MESA data can add cardiovascular and biomarker variables

Why it is useful:

- strong if sleep is central to the simulator
- objective sleep measurements are more informative than only self-report

Main limitation:

- not as simple as NHANES for a first pass
- broader MESA exam variables may require additional access routes

Source:

- https://sleepdata.org/datasets/mesa

## Sleep/Cardiovascular Candidate: SHHS

Useful variables:

- sleep-disordered breathing measures
- polysomnography
- cardiovascular outcomes and hypertension-related variables
- adults aged 40+

Why it is useful:

- good for sleep and cardiovascular aging questions

Main limitation:

- less suited for eating behavior
- older cohort and sleep-specific design

Source:

- https://sleepdata.org/datasets/shhs

## Broad Modern Candidate: All of Us

Useful variables:

- demographics
- surveys
- physical measurements including blood pressure, heart rate, BMI, height, weight, waist/hip circumference
- EHR measurements
- Fitbit-derived heart-rate, activity, and sleep summaries for participants with wearable data
- biospecimens/genomics in controlled tiers

Why it is useful:

- broad, modern, and multi-modal
- useful later for realistic measurement patterns

Main limitation:

- access/workbench overhead
- EHR and wearable missingness can be complicated

Source:

- https://support.researchallofus.org/hc/en-us/articles/4619151535508-Data-Types-and-Organization

## Older Adult Aging Candidate: HRS

Useful variables:

- adults over 50
- health and functioning
- blood-based biomarkers
- psychosocial data
- longitudinal structure

Why it is useful:

- directly focused on aging
- good for later biological-aging framing

Main limitation:

- less ideal for simple diet/sleep/fitness/blood-pressure simulator than NHANES
- older age range only

Source:

- https://odphp.health.gov/healthypeople/objectives-and-data/data-sources-and-methods/data-sources/health-and-retirement-study-hrs

## Simple Behavior Survey Candidate: BRFSS

Useful variables:

- age
- physical activity
- fruit and vegetable consumption
- self-reported health conditions and behaviors
- optional modules can include additional topics

Why it is useful:

- very large and public
- easy behavioral variables

Main limitation:

- mostly self-report
- lacks direct clinical measurements like measured blood pressure and labs
- less suitable for a biological-age simulator unless used only for behavior priors

Source:

- https://odphp.health.gov/healthypeople/objectives-and-data/data-sources-and-methods/data-sources/behavioral-risk-factor-surveillance-system-brfss

## Practical Recommendation

Start with NHANES.

First simulator grounding:

```text
chronological age -> simple bioindicator distributions
```

Candidate first indicators:

- systolic blood pressure
- diastolic blood pressure
- BMI or waist circumference
- sleep duration
- physical activity score or sedentary time
- diet proxy such as fruit/vegetable intake or selected nutrient intake
- resting heart rate or fitness proxy if available for the chosen cycle

Then add realism step by step:

1. fit marginal age trends for each indicator
2. estimate residual variance by age group
3. add residual correlations between indicators
4. add sex as a covariate
5. add missingness/noise patterns

Do not claim this is true biological age at first. Treat it as a simulator for age-conditioned health indicators. A later model can introduce a latent biological-age offset after the measurement model is stable.

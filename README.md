# DMT_Assignment_2

--- Due: ?

Task 1 - Margherita

Task 2 - Avril

Task 3 - Milou

--- Due: 6 May

Task 4 - All 

--- Due: 13 May

Task 5 (Ethical AI) - Margherita

Task 5 (Scalable Deployment) - Milou

Process Report - Avril

+ all write about their Task 4 models

# Leakage removal

Issue:
- 'position' is a target variable (y), so we cannot use it as a feature (X) that goes into the model for prediction.
- 'prop_bool_rate' and 'prop_click_rate' also depend on 'position', so we cannot use these features either.
- Since missing values for 'position' are replaced with -1 and CatBoost automatically replaced missing values with the minimun value for each feature and we didn't include this feature in KNN or LightGBM, no error was raised!

Solution steps:
1. Remove 'prop_bool_rate', 'prop_click_rate', and 'prop_location_score2_isnull'. Since this is the only engineered features we have, we'll have to think of other ones.
2. Re-run LightGBM and CatBoost with the leakage-free features (no position, click_bool, booking_bool, or gross_booking_usd)
--> Due this Sunday 17 May
3. Re-run KNN + re-run the bias correction
4. Adjust Overleaf accordingly
--> Due next Sunday 24 May

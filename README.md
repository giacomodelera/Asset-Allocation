# Asset Allocation

# AI for Blood Cell Identification

## Abstract
This study examines portfolio allocation strategies for the S&P500 index, which contains 11 sector indices and 5 factor indices. 

The analysis focuses on creating efficient portfolios using 2023 financial data through different techniques like the typical efficient frontier, constrained optimization, Black-Litterman model, and Principal Component Analysis. 

Portfolios are compared to an equally weighted benchmark using performance metrics such as risk, return, and diversification.
The in-sample analysis reveals the best allocations under various constraints, while the out-of-sample evaluation over 2024 analyzes the stability of these approaches. The results emphasize
the trade-offs between risk and return and the impact of constraints on portfolio diversification.
This study shows how portfolio theory is actually applied under dynamic market conditions.

## Repository Structure
* [Report](Report.pdf)
  contains a deeper description of the problem and a detailed explanation of the methodologies.
* **`Code/`**
  
  **`1_dataset_augmentation`** contains code for augmenting the dataset by applying various transformations (e.g., rotation, flipping, scaling) to the images.
  
  **`2_model_training`** is responsible for training the AI model. It loads the augmented dataset, sets up the deep learning architecture, and trains the model using the appropriate loss function and optimization     technique.
  
  **`3_zip_for_submission`** is used to format the model and weights in order to easily submit results to the competition platform.



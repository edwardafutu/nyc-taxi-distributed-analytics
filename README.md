# Spark-Based Analysis of NYC Yellow Taxi Trip Records

## Project Overview

This project applies distributed data processing, statistical analysis, regression modelling, and machine learning to New York City Yellow Taxi trip records for January 2026.

The project was completed as part of **BDAT 608: Computational Statistics in Big Data II**, during the MSc Big Data Analytics programme at the University of Ghana.

Using R and Apache Spark, we analysed millions of taxi trip records to examine relationships between trip characteristics, passenger tipping behaviour, and payment methods.

## Project Objectives

The main objectives were to:

- Process and analyse large-scale taxi trip data using Apache Spark.
- Clean the raw dataset and engineer relevant trip characteristics.
- Examine relationships among numerical variables using correlation analysis.
- Develop simple and multiple linear regression models to investigate recorded tip amounts.
- Build a reusable R function to automate Spark-based regression workflows.
- Develop classification models to predict whether passengers paid by card or cash.

## Dataset

The dataset was obtained from the New York City Taxi and Limousine Commission (TLC).

- **Dataset:** Yellow Taxi Trip Records
- **Period:** January 2026
- **Original size:** 3,724,889 records
- **Original variables:** 20
- **Format:** Parquet

The data include trip distance, pickup and drop-off times, passenger count, fare amounts, tips, payment method, and other trip-related information.

Following data cleaning and feature engineering, approximately 2.46 million records were retained for analysis.

## Tools and Technologies

- **R** – Data preparation, statistical analysis, modelling, and visualisation.
- **Apache Spark** – Distributed data processing and machine learning.
- **sparklyr** – Connecting R to Apache Spark.
- **Arrow** – Reading and writing Parquet datasets.
- **dplyr** – Data manipulation and transformation.
- **tidymodels** – Statistical and machine learning modelling.
- **ggplot2** – Data visualisation and diagnostic plots.
- **corrr** – Correlation analysis.

## Methodology

The project was organised into the following stages:

1. **Data loading and preparation:** Loaded the Parquet dataset, examined data quality, removed invalid records, and engineered variables such as trip duration, average speed, pickup hour, weekend status, and airport-trip indicators.

2. **Distributed data processing:** Established a local Apache Spark environment and transferred the cleaned dataset into Spark for analysis.

3. **Correlation analysis:** Compared correlation matrices generated using Base R, Spark, and the `corrr` package. A correlation heatmap was also produced to examine relationships among numerical variables.

4. **Regression modelling:** Developed simple and multiple linear regression models to examine the relationship between trip characteristics and recorded tip amounts. Model diagnostics were also examined.

5. **Automated regression function:** Developed a reusable R function, `spark_regression()`, to automate model fitting, coefficient extraction, performance summaries, prediction generation, and residual diagnostics.

6. **Classification modelling:** Implemented logistic regression, boosted tree, and random forest models using the Spark engine to classify passenger payment methods.

## Selected Findings

- Trip distance and fare amount showed strong positive correlations with total trip charges.
- The multiple linear regression model explained approximately 43% of the variation in recorded tip amounts.
- The boosted tree classifier achieved approximately 88.4% accuracy using the full predictor set.
- Classification accuracy was affected by the imbalance between card and cash transactions, highlighting the need for additional class-specific evaluation metrics.

An important data limitation is that cash tips are not recorded in the TLC dataset. Consequently, recorded tip amounts should not be interpreted as the actual tips received by drivers across all payment methods.

## Skills Demonstrated

- Distributed data processing
- Large-scale data cleaning and feature engineering
- Statistical correlation analysis
- Linear regression and model diagnostics
- Machine learning classification
- R function development and workflow automation
- Working with Parquet datasets and Apache Spark

## Contributors

This was a collaborative group project completed by:

- Albert Jahsking Viala
- Samuel Opoku
- Edward Afutu Amoni
- David Kwame Tsigbey
- Degraft Kofi Obeng

## Author

**Edward Afutu Amoni**

MSc Big Data Analytics, University of Ghana

[GitHub](https://github.com/edwardafutu) | [LinkedIn](YOUR-LINKEDIN-URL)

## Data Source

New York City Taxi and Limousine Commission (TLC): [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

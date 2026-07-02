# Football Performance Analytics: Premier League 23/24 EDA

## Overview
This project presents an in-depth Exploratory Data Analysis (EDA) of the English Premier League 2023/2024 season. The primary goal was to extract actionable business and performance insights from raw match data (380 matches) by combining statistical analysis with custom-built metrics. 

While the dataset revolves around sports, the data-handling methodologies applied here (data cleaning, transformations, correlation analysis, and feature engineering) are highly transferable to financial and macroeconomic market analysis.

## Tech Stack
* **Language:** Python
* **Libraries:** `Pandas`, `NumPy`

## Key Analytical Features & Feature Engineering
Rather than relying solely on provided data points, this project introduces custom performance indicators:
* **Aggression Index:** A weighted metric calculating the "dirtiness" of a team's playstyle based on the ratio of fouls to cards.
* **Cost of a Point:** An efficiency metric evaluating how many goals a team needs to score, on average, to secure a single league point (highlighting over/under-performing teams).
* **Home Advantage Analysis:** A dumbbell chart visualization proving the statistical advantage of home matches on Expected Goals (xG) generation.

## Key Findings
* Teams like Liverpool and Newcastle showed a massive variance in xG when playing at home vs. away, while Manchester City maintained a stable xG regardless of the venue.
* A strong correlation was found between the number of goals conceded and the frequency of fouls committed, suggesting tactical shifts or loss of game control.
* The most frequent exact scoreline across the entire season was 1:1.

## How to view
To view the full report with interactive visualizations, download or open the `Projekt_EDA.html` file in your browser.

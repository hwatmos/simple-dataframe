# Simple DataFrame

Simplistic replacement of pandas

## Vision

My vision is for this library to be the most trusted Python library for data science, empowering organizations to confidently install and utilize a data science library on their local assets. By maintaining transparency through the exclusive use of pure Python and standard libraries, I ensure that my code is easily analyzed for vulnerabilities. Unlike other libraries with complex dependencies, my commitment to simplicity and security enables organizations to perform essential data science tasks without compromising safety.

While my library does not aim to replace more powerful libraries, it provides sufficient functionality for local data manipulation. I envision data scientists using this library to prepare and transform data locally, with the ability to easily encode, anonymize, and export this data to cloud environments for more advanced analysis. In this way, I enable secure and efficient data handling, ensuring that data is properly transformed before entering more complex and resource-intensive analytical workflows.

## Goals

1. Trustworthiness and Security

Ensure the library is safe and trustworthy for organizational use by utilizing only pure Python and standard libraries. This approach facilitates easy review by cybersecurity specialists.

2. Functionality and Integration

Provide essential data science functionalities for local data preparation and export for easy ingestion into more powerful libraries (like scikit-learn, statsmodels, pandas) in a cloud environment.

3. Transparency

Develop the library with transparent, understandable code, avoiding non-standard Python libraries. Provide thorough yet concise documentation, both functional and technical.

# Core Functionality

The library is capable of:

* Loading two-dimensional data
* Applying manipulations to columns
* Grouping and aggregating
* Joining and concatenating datasets
* Creating dummy variables
* Exporting data

Additionally, it can completely anonymize a dataset while remembering all anonymization rules internally. The library's main class can be considered a processor rather than a data frame, focusing on data transformation rather than on the data itself.

## Functionality

**Tto-dos**
* Encode columns and remember encoding rules within the object itself.
* Get dummies but remember which dummy cols describe the same feature.
  * This way don't need to display all dummies, can just display the original column with an indicator that it has been either "one-hot'ed" or "dummied" etc.
* Concat
 
**Specific to strings**
* Common operations on strings that pandas makes accessible via .str?

**Specific to dates**
* Need to plan this out
* First of the month
* str to date

**Loosely or un-defined**
* Simplify the common task of specifying dict of col labels and value types.
* Get unique values. I use this a lot so maybe just implement functionality like df.col_name and it will print out unique if it is a categorical col?
* Need a simplified mechanism for group by. I do this all the time, maybe specify each column as either Dimension or Measure? Then I can create method for collapsing dimensions but measure recalc may atipp be dependent on other measures and even in thr dimensions...

# Research

* Descriptors
  * https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    * https://docs.python.org/2/reference/datamodel.html#implementing-descriptors
    * https://eev.ee/blog/2012/05/23/python-faq-descriptors/
* Typing
  * https://docs.python.org/3/library/typing.html
  * https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
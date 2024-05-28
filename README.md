# Simple DataFrame

Simplistic replacement of pandas

# Vision

Pandas has been great but performing data science requires A LOT of knowledge and thus has a very steep curve for those who are just starting out with Python or data science.  In addition, pandas provides much more functionality than just strict data-science tasks, which makes it even more difficult to learn essential data science tasks.  This package aims to address these issues.  In addition, this pckage is being developed by a DoD employee with the hopes of making it easier for other DoD Data Scientists to enable data science functionality.

The main means of accomplishing a simplified data science workflow is by designing a "data frame" class which stores details of data transformation workflows (more on that in Functionality).

## Functionality

**Important to-dos**
* Check for duplicate column labels.
* 

**Well defined**
* Encode columns and remember encoding rules within the object itself.
* Get dummies but remember which dummy cols describe the same feature.
  * This way don't need to display all dummies, can just display the original column with an indicator that it has been either "one-hot'ed" or "dummied" etc.
* Change column types easily.
* Column.isna()
* Check NAs
* Replace NAs with replacement rules
* Merge (and specific method for left_join)
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
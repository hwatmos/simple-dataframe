# Simple DataFrame

Simplistic replacement of pandas

# Vision

## Functionality

**Well defined**
* Encode columns and remember encoding rules within the object itself.
* Get dummies but remember which dummy cols describe the same feature.
  * This way don't need to display all dummies, can just display the original column with an indicator that it has been either "one-hot'ed" or "dummied" etc.
* Change column types.
* 
 
**Specific to strings**
* Common operations on strings that pandas makes accessible via .str?

**Specific to dates**
* Need to plan this out
* First of the month
* str to date

**Loosely or un-defined**
* Simplify the common task of specifying dict of col labels and value types.

# Research

* Descriptors
  * https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    * https://docs.python.org/2/reference/datamodel.html#implementing-descriptors
    * https://eev.ee/blog/2012/05/23/python-faq-descriptors/
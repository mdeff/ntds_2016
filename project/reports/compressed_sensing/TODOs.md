# TODOs

## Structure reminder
  1. **Data acquisition:** from the web, a database, a flat file, etc. This includes cleaning the data.
  2. **Data exploration:** simple exploratory analysis to describe what you got.
  3. **Data exploitation:** build and train a Machine Learning algorithm based on this data. Any algorithm is considerable, but it has to be motivated.
  4. **Evaluation:** evaluate the performance, e.g. accuracy, training time, etc., of the chosen model. You define the metrics you care about! If you tried multiple algorithms, please report their performance and try to explain it.


## Data acquisition
  - Try to download the data set from the web instead of having it hosted on github
    - Link to main page: http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases
    - Link to direct download (takes some time to download): http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/standard_test_images.zip

### Data cleaning
  - Convert to greyscale -> original images
    - **Should it be rescaled to [0, 255] when generating the orig dataset folder??**, *e.g.* airplane is not whereas arctichare is
  - **All images should have the same resolution???**
  - images as a dictionary `dict()`?
  - Split in patches of size 32 x 32 (to be generalized)
  - Center the data
  - Measurement matrix:
    - Gaussian
    - Bernoulli
  - Fixing the seed for reproducible results
    - Check if should be in a file, for now it is defined in the `utils.create_measurement_model()` function


## Data exploration
  - Figure showing an image example, a specific patch and its compressed version (*i.e.* mixed + blurred)

## Data exploitation


## Evaluation

## Installation
  - Check a docker installation on the NTDS Moodle
  - check_instal.py for manual installation

## License

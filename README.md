# UCD-Thesis
## Introduction
Geocoding refers to the process of converting physical features, for example street addresses given in natural language to spatial coordinates on the Earth's surface. Geocoding has a number of applications, the most well known of which is in the field of Geographic Information Systems (GIS) and Spatial Analysis. GIS applications include mapping and navigation services such as Google Maps, telecom services, urban and transportation planning, banking and taxation, amongst many others (nobelsystemsblog.com). Furthermore, Geocoding has been used in Augmented Reality products like Pokemon Go. It plays an essential role in developing localised marketing strategies by contributing to geographic segmentation. Its importance is especially highlighted in its ease of use in products like Google Maps where the user just needs to provide the target address in natural language and the application automatically converts that into location coordinates and provides appropriate directions to the user.
<br/>This process involves several steps: from reading and identifying parts of the street addresses to correctly matching them to unique locations and computing their locational coordinates in terms of latitude and longitude.
<br/>While this may seem like a simple operation to a non-technical user of the aforementioned applications, as we will discover in the next few pages, it involves some fairly complex algorithms and statistical models.
<br/>We define the problem of street address classification as follows:
<br/>Given a street address: "Langdale Lodge, Main Street, York, Harrogate, North Yorkshire", we want the output of our statistical model to be:
<br/>House/Apartment: Langdale Lodge
<br/>Street: Main Street
<br/>City/Town: York
<br/>District: Harrogate
<br/>County: North Yorkshire
<br/>The problem of street address classification falls under the umbrella of Natural Language Processing, more specifically, Named Entity Recognition. Identifying parts of a street address is akin to assigning each word in the address a tag that appropriately describes its class. In technical terms, this translates into building a customised Parts of Speech tagger for street addresses that correctly classifies each word in the address. Street address classification is an initial stage in the overall process of Geocoding where the address is tokenised and appropriately classified and can then be used in the next stages of the process. In this paper, we will tackle the  problem of identifying parts of street addresses using generative and discriminative statistical models and compare their performance using multiple performance metrics.
<br/><br/><br/>
## Data and Models
The csv file named addresses_month.csv contains data extracted from the UK government website. All other csv files have been the result of processing and standardising cleaned_addresses.csv.
<br/> <br/>1. standardise_data.py
<br/> This file contains code that normalises the strings given in the raw dataset as follows:
<br/>	All strings were converted to lowercase
<br/>	Trailing spaces and unnecessary spaces in the middle of the strings were removed
<br/>	All punctuation marks were removed
<br/>	stopwords from the address were removed using the nltk package
<br/>	All non-ascii characters were removed
<br/>	All strings containing at least one number were converted to strings containing uppercase D of the same length
<br/>	Both SAON and locality columns were dropped
<br/>	<br/>	2. preprocessing.py
<br/> The code in this file pre processes the data by cleaning it appropriately and getting it in a format that is ready for analysis.
<br/>	<br/>	3. structured_perceptron.py
<br/> This file contains the code for HMM and the Structured Perceptron.
<br/>	<br/>	4. recurrent_neural_network.py
<br/> This file contains the code for the recurrent neural network.

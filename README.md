# lasagne-caterer
Making lasagne can be tedious. why not just cater when you want italian goodies?

Lasagna making has some components:

* A _recipe_ for putting together the raw lasagne by mixing and matching layers. 
    * Classes for sequentially applying lasagna layers. Result should be compiled theano functions.
* An _oven_ to prepare the lasagna for use it for use. (raw lasagnas are useless)
    * Classes for reading in data and feeding it to the lasagna == heating. 
    A raw lasagna needs a lot of heat (training data), whilst a lasagna cold from the fridge just need reheating (test data)
* A _fridge_ for conserving the prepared lasagna for later use
    * Classes that enables saving and loading lasagne models
* A _cook_ to orchestrate events such that several lasagnas can be made.
    * Classes that glue together any _recipe_, _oven_ and _fridge_ to make one or more lasagnas
* A _menu_ that provides easy access to useful combinations of the above classes
    * Classes for combining specific combinations that are known to yield good results
    
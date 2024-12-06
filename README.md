# ECE661_project
Class project for ECE661 at Duke 

# Files

### lab.py 
This is the file to run, you can use the hardcoded values or command line arguements (look at utils/parse_args for all the arguements)

### data.py
Controls creation and formatting of dataset 

### mamba.py & pscan.py
Declarations of MambaStock model from [MamabaStock repo](https://github.com/zshicode/MambaStock)

### models.py
Controls all the model types (MambaStock, LSTM, Transformer)

### attacks.py
Enables two different kinds of membership inference attacks, 1 that simply computes loss target models loss of a possible stock and 1 that trains and uses shadow models to train a membership classifier

### utils.py
Random Utility functions for lots of things

### symbols.txt
List of stock symbols to be used in the initial training (and testing)

### graph.txt
Single stock symbol to graph 

### all_symbols.txt
A superset pf stock symbols used to evaluate the success of membership inference attacks

# TODO
- Make a better README
- Probably other things

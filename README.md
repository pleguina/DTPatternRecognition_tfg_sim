# Pattern Training

  Code for plotting pseudo-bayes and STD patterns

## Instructions (Work in progress)
### How to use the code

  The code works reading inputs from a log file. By default, the code will 
  try to read a csv file named ``EventDumpList_StdToBayes.csv``, which has been
  generated from the .log file that is obtained after running the pattern training
  with CMSSW.

  In order to run the code you just huve to run ``python main.py -s {station}``, where
  station can be either MB1, MB2, MB3 or MB4. Then, the command line
  will prompt you with the following line:

  `What patterns do you want to plot? Note that you can either give a single index, or a comma-separated list of patterns`

  Then, you can go to the csv file, find the muon that you want to plot, and enter the number in the first column (which
  is an identification number to know which pattern are you plotting) and then a matplotlib canvas will appear. 
  The prompt will keep appearing until you type "exit". Each time you enter patterns, the canvas will be overwritten with
  the new ones.


### About the log files
  The only difference between the csv file and the .log file (both are in the repository)
  is that, if each line represents a muon pattern, then the element in the first column
  represents the identification number that you need to enter in the command line.
  

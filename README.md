# Website Fingerprinting: A QUIC look
Website Fingerprinting attacks aim to identify the websites a user visits, merely from its the encrypted traffic. While most research works focused on TCP as the transport protocol, the ongoing shift to the newer transport protocol QUIC calls for a study of the website fingerprinting attacks in the QUIC era. Three deep learning models are used to evaluate the success of website fingerprinting attacks on the QUIC protocol, namely [Var-CNN](https://github.com/sanjit-bhat/Var-CNN), [Deep Fingerprinting](https://github.com/deep-fingerprinting/df) and [LSTM](https://github.com/DistriNet/DLWF)).

The folders are organised as follows:
<ol>
  <li>`Data Processing` contains all scripts to process and clean the data</li>
  <li>`Experiments` contains the code for all our experiments to generate the results</li>
 </ol>

## Dataset
The traces for our experiment are available [here](https://mega.nz/folder/pZdgCChS#E1JMxCFwzI5wtaDlw45QYA). They need to be cleaned for easy reading via libraries such as [pandas](https://pandas.pydata.org/). To do so, run automate_clean_data with both tar files in the git root directory. This cleans, extracts the data. The data can be found in Data_Processing after the process is finished.

The Experiments folder contains code for the different evaluations done, as well as the models used and the auxillary functions such as read data.

## Contact
For any queries, please feel free to raise issues or contact the authors.

## References
TBA

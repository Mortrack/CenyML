

WHY I DID NOT EVALUATE PERFORMANCE AND ONLY VALIDATED OBTAINING CORRECT RESULTS WITH THE "QUICK SORT" CENYML METHOD?


This method was only validated to give correct results but it was not validated in terms of performance.
This is because the processing time of the quick sort method is highly dependent on the actual
arrangement of the data in the input matrix where its best case time complexity is O(n*log(n)) and its
worst one is O(n^2). Due to this fact, a fair performance evaluation between the CenyML method and the
NumPy method would be to evaluate for every possible permutation case of the input matrix. Now, the thing
is that this can be relatively quickly done in C under the CenyMl method, but on Python we would have to
wait for a crazy amount of time (several hours or days) just to get one sample result due to the difference
of processing time between C and Python when obtaining the permutations for each case of the input matrix.
If there is a way to obtain comparable or faster processing times for the permutation process in Python, at
least i was not able to obtain them neither writing RAW Python code or the NumPy library.

In conclusion, it was determined not to evaluate the performance on the quick sort method because i could
not afford to spend waiting so much time to obtain each sample by permutation the input matrix in Python
and because the quick sort method is not a machine learning method or strictly required in them, considering
that the CenyML focuses on the machine learning methods primarly. However, the best attempt was made to
obtain the fastest possible results with respect to the knowledge and experience that i have as a programmer.




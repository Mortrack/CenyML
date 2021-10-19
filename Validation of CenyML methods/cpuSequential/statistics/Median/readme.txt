

WHY I DID NOT EVALUATE PERFORMANCE AND ONLY VALIDATED OBTAINING CORRECT RESULTS WITH THE "QUICK MEDIAN" CENYML METHOD?


This method was only validated to give correct results but it was not validated in terms of performance.
This is because the processing time of the quick sort method is highly dependent on the actual
arrangement of the data in the input matrix where its best case time complexity is O(n*log(n)) and its
worst one is O(n^2). Due to this fact, a fair performance evaluation between the CenyML method and the
NumPy method would be to evaluate for every possible permutation case of the input matrix. Now, the thing
is that this can be relatively quickly done in C under the CenyMl method, but on Python we would have to
wait for a crazy amount of time (several hours or days) just to get one sample result due to the difference
of processing time between C and Python when obtaining the permutations for each case of the input matrix.
If there is a way to obtain comparable or faster processing times for the permutation process in Python, at
least i was not able to obtain them neither writing RAW Python code or the NumPy library. In addition, there
seems to be a variant method called "quick median" in which you directly search for the sorted index position
that you desired based in the "quick sort" method, instead of sorting every single value like i did. This
variant method seems to be the one to be used in NumPy and it seems to be way faster than the traditional
median method in which you have to first sort all the possible values. However, i was not able to replicate
such method for now.

In conclusion, it was determined not to evaluate the performance of the quick median method because i could
not afford to spend neither waiting so much time to obtain each sample in Python by permuting the input
matrix; i could not spend more time in trying to replicate the "quick median" method as applied in NumPy and
because the quick median method is not a machine learning method or strictly required in them, considering that
the CenyML focuses on the machine learning methods primarly. However, the best attempt was made to obtain
the fastest possible results with respect to the knowledge and experience that i have as a programmer.




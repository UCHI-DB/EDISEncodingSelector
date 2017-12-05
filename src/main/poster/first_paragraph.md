# Data-Driven Lightweight Encoding Selection
This project explores an automated method to select efficient lightweight encoding schemes for column-based databases, such as run length and bitmap encoding. Comparing to popular compression techniques like gzip and snappy, lightweight encoding has the following benefits:
* Speed: Lightweight encodings involve only simple operations and usually has negligible impact on data access time. 
* Local Computation: Most Lightweight encoding schemes only access adjacent local tuples during encoding/decoding process, without the need to go through entire dataset.
* In-Site Query Execution: Encoding schemes enable in-place query execution that allows queries to be directly applied on encoded data and further reduce the time needed for query processing, instead of requiring a large block of data to be uncompressed first.

However, there is no simple rule on how to choose the best encoding scheme for a given attribute. Existing systems either rely on database administrators' personal experience or use simple rules to make selection. In practice, neither of these two methods achieve optimal performance. Our method attacks the problem in a systematic data-driven way. Our effort includes:
* Dataset Collection
* Pattern Mining
* Data-Driven Encoding Prediction
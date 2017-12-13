# Data-Driven Encoding Selector
This project explores an automated method to select efficient lightweight encoding schemes for column-based databases, such as run length and bitmap encoding. Comparing to popular compression techniques like gzip and snappy, lightweight encoding has the following benefits:
* Speed: Lightweight encodings involve only simple operations and usually has negligible impact on data access time. 
* Local Computation: Most Lightweight encoding schemes only access adjacent local tuples during encoding/decoding process, without the need to go through entire dataset.
* In-Site Query Execution: Encoding schemes enable in-place query execution that allows queries to be directly applied on encoded data and further reduce the time needed for query processing, instead of requiring a large block of data to be uncompressed first.

However, there is no simple rule on how to choose the best encoding scheme for a given attribute. Existing systems either rely on database administrators' personal experience or use simple rules to make selection. In practice, neither of these two methods achieve optimal performance. Our method attacks the problem in a systematic data-driven way. Our effort includes:
* Dataset Collection
* Data-Driven Encoding Prediction
 
## Dataset Collection
Our work is based on analysis and evaluation of real-world datasets. We have created an automated framework to collect datasets, extract columns from them, organize, and persist the records for further analysis. The framework could accept various input formats including csv, txt, JSon and MS Excel files. It also supports recognition of column data type for unattended data collection. For further analysis purpose, the framework provides API enabling customized features to be extracted from the columns.

Using this framework, we have collected over 7000 columns from approximately 1200 datasets with a total size of 500G data. These datasets are all from real-world data sources and cover a rich collection of data types (integer, date, address, etc.), with diverse data distributions. We use Apache Parquet's built-in encoders to encode these data columns with different encoding schemes, looking for the one performing best for each column. We also developed some customized encoders to compare their performance.

## Data Driven Encoding Prediction

In this project, we target at building a encoding selector that is able to "predict" the best encoding scheme by looking at only a limited section of the entire dataset, e.g., the first 1M records. To do this, we use the collected data columns and the "ground truth" best encoding scheme to train a neural network to make prediction. We modify Apache Parquet code to support per-column encoding setting, and inject our encoding prediction code to choose best encoding for a given data column on the fly. Besides storage size, we also study how different encodings have impact on query performance.

## Encoding tool based on Apache Parquet

As an example, we provide a command line tool to encode a given CSV file into Apache Parquet format while automatically choose encoding for each column. The source code can be found at *src/main/scala/edu/uchicago/cs/encsel/app/encoding/ParquetEncoder*

User should invoke the tool as following:

**ParquetEncoder <input_file> <schema_file> <output_file> <int_model> <string_model>**

where <input_file> is a CSV file to be encoded, <schema_file> describes the name for each column. The encoded file will be written to <output_file>.

<int_model> and <string_model> are pre-trained models for integer type and string type. We currently only support encoding selections for these two types as the number of encodings for other data types are limited.

We also provide a Python script to train models from data. The script can be found at
*src/main/pythonencsel/train.py*

User should invoke the script as following:

**train.py <data_file> <model_path> <num_class>** 

<data_file> is a CSV file contains features extracted from data columns and the preferred encoding. Refer to *Features* for available features. Currently we use 21 features, so the CSV file should contain 22 columns, where the last column indicates encoding type.

<model_path> is the path to write model file to.

<num_class> is the number of encoding types for the given data file. Integer type supports 5 encoding types, and string type supports 4 types. For details of encoding type supported, refer to *IntEncoding* and *StringEncoding*

We include pre-trained models for both string and integer data type in the source code distribution. User can find them under folder *src/main/model*
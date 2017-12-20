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

## Feature Selection
We choose 5 categories, 19 features to be used in prediction. We brief them here and the related code can be found in package *edu.uchicago.cs.encsel.dataset.features* 

* **Sparsity**
    * **valid ratio** Number of non-empty records vs. number of total records
* **Entropy**
We compute the Shannon entropy of characters in each line, and collect the statistical info 
    * **line min**
    * **line max**
    * **line mean**
    * **line variance**
* **Length** The number of characters in each line
    * **min**
    * **max**
    * **mean**
    * **variance**
* **Distinct** Cardinality of the column
    * **ratio** Number of distinct records vs. number of total records
* **Sortness** We use a sliding window to compute the number of inverse pairs in order to measure
how much "sorted" the records are. We choose sliding window size of 50, 100 and 200, and compute three measures. Kendall's Tau, Spearman's Rho and Inverse Pair.
    * **inv pair** = abs(1 - (inverted pair / total pair))
    * **kendall's tau** See https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    * **spearman's rho** See https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    

## Data Driven Encoding Prediction

In this project, we target at building a encoding selector that is able to "predict" the best encoding scheme by looking at only a limited section of the entire dataset, e.g., the first 1M records. To do this, we use the collected data columns and the "ground truth" best encoding scheme to train a neural network to make prediction. We modify Apache Parquet code to support per-column encoding setting, and inject our encoding prediction code to choose best encoding for a given data column on the fly. Besides storage size, we also study how different encodings have impact on query performance.

## Encoding tool based on Apache Parquet

### Parquet File Encoder
As an example, we provide a command line tool to encode a given CSV file into Apache Parquet format while automatically choose encoding for each column. The source code can be found at *src/main/scala/edu/uchicago/cs/encsel/app/encoding/ParquetEncoder*

User should invoke the tool as following:

**ParquetEncoder <input_file> <schema_file> <output_file> <int_model> <string_model>**

where <input_file> is a CSV file to be encoded, <schema_file> describes the name for each column. The encoded file will be written to <output_file>.

<int_model> and <string_model> are pre-trained models for integer type and string type. We currently only support encoding selections for these two types as the number of encodings for other data types are limited. To train a customized model based on your own data, please check the next two sections for feature extraction and model training.

### Feature Extraction
We provide a toolchain to extract features from raw data files. It currently supports csv, tsv and excel formats.

**edu.uchicago.cs.encsel.dataset.GuessSchema <folder_path>** 
will generate Parquet schema files for data files in the path. This requires a thorough scan of data files and may take some time.

**edu.uchicago.cs.encsel.dataset.CollectData <folder_path>**
will read the schema file and extract each data column into separate files. The generated files will be stored in path *<ENC_ROOT>/columns*, and metadata such as data type, file path will be persisted in either a metafile or database. This behavior can be configured by modify the default implementation in *edu.uchicago.cs.encsel.dataset.persist.Persistence*

**edu.uchicago.cs.encsel.dataset.CollectFeature** will read the metadata, and extract features as mentioned in section *Feature Selection* from all the columns. The extracted features will be persisted the same way as column metadata. User can extract them later for model training.


### Model Training

We provide a Python script to train models from collected features. The script can be found at
*src/main/pythonencsel/train.py*

User should invoke the script as following:

**train.py <data_file> <model_path> <num_class>** 

**<data_file>** is a CSV file contains features extracted from data columns and the preferred encoding. Refer to *Feature Selection* for available features. User can use the toolchain described in previous section to collect feature values from data files.
 
 Currently we use 19 features, so the CSV file should contain 20 columns, where the last column indicates preferred encoding type.

**<model_path>** is the path to write model file to.

**<num_class>** is the number of available encoding types for this training session. Integer type supports 5 encoding types, and string type supports 4 types. For details of encoding type supported, refer to *IntEncoding* and *StringEncoding*

We include pre-trained models for both string and integer data type in the source code distribution. User can find them under folder *src/main/model*
## **Predict Research Significance Based on Its Abstract**

#### A. Data Collection  
Raw data were fetched from [Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/) in Medline format with different start date, and were divided into three groups:  
  * **Group 1** includes publications from **20** journals:  
  Nature (from: 2017-01-01),<br>Science (from: 2017-01-01),<br>Cell (from: 2017-01-01),<br>New England Journal of Medicine (nejm) (from: 2015-01-01),<br>Nature Genetics (nat_genet) (from: 2015-01-01),<br>Nature Biotechnology (nat_biotechnol) (from: 2015-01-01),<br>Nature Medicine (nat_med) (from: 2015-01-01).<br>
  Cancer Cell (from: 2015-01-01),<br>
Nature Immunology (nat_immunol) (from: 2015-01-01),<br>
Cell Metabolism (cell_metab) (from: 2015-01-01),<br>
Cell Stem Cell (from: 2015-01-01),<br>
Nature Cell Biology (nat_cell_biol) (from: 2015-01-01),<br>Nature Methods (nat_methods) (from: 2015-01-01),<br>Nature Structural & Molecular Biology (nat_stru_mol_biol) (from: 2015-01-01),<br>Nature Neuroscience (nat_neuro) (from: 2015-01-01),<br>
Molecular Cell (mol_cell) (from: 2015-01-01),<br>Immunity (from: 2015-01-01),<br>Journal of Clinical Oncology (jco) (from: 2017-01-01),<br>Gut (from: 2015-01-01),<br>Gastroenterology (gastro) (from: 2017-01-01).<br>

* **Group 2** includes publications from **15** journals: <br>
Neuron (from: 2016-01-01),<br>The Journal of Clinical Investigation (jci) (from: 2017-01-01),<br>Hepatology (from: 2017-01-01),<br>Blood (from: 2017-01-01),<br>Genes & Development (gd) (from: 2015-01-01),<br>Developmental Cell (dev_cell) (from: 2015-01-01),<br>Elife (from: 2017-01-01),<br>Proceedings of the National Academy of Sciences of the United States of America (pnas) (from: 2017-09-01),<br>PLOS Biology (plos_biol) (from: 2016-01-01),<br>Current Biology (cb) (from: 2017-01-01),<br>Nucleic Acids Research (nuc_acid_res) (from: 2017-06-01),<br>The EMBO Journal (embo_j) (from: 2016-01-01),<br>Cell Research (cell_res) (from: 2016-01-01),<br>Nature Communications (nat_comm) (from: 2018-01-01),<br>Plos Medicine (plos_med) (from: 2015-01-01).

* **Group 3** includes publications from **11** journals: <br>
Science Signaling (sci_signal) (from: 2015-01-01),<br>Journal of Biological Chemistry (jbc) (from: 2017-01-01),<br>Scientific Reports (sci_rep) (from: 2018-02-01),<br>Plos One (from: 2018-02-01),<br>Development (from: 2016-01-01),<br>Developmental Biology (dev_biol) (from: 2016-01-01),<br>Molecular and Cellular Biology (mcb) (from: 2015-01-01),<br>Journal of Cell Biology (jcb) (from: 2016-01-01),<br>
Oncotarget (from: 2018-01-01),<br>Journal of Molecular Biology (jmb) (from: 2016-01-01),<br>Oncogene (from: 2016-01-01).


#### B. Data Munging
* Downloaded medline formatted data from each journal were converted to pandas dataframe type with columns: **'PMID', 'Title', 'Abstract'** and **'Journal'**. Detailed code could be found in ***format_convert.py***.<br>

* All converted data in the same group were combined into one single dataframe. Records with incorrect data in **'Abstract'** column were removed. Detailed code could be found in ***data_combine.py***<br>

* Summary of the cleaned data were shown below:<br>
    * Total data in all groups:<br>
    ![](total_count.jpg)<br>
    * Counts in individual groups:<br>
    ![](group_1_counts.jpg)
    ![](group_2_counts.jpg)
    ![](group_3_counts.jpg)<br>

#### C. Feature Engineering
* Data split:
 1. Data in each group were randomly shuffled by rows;
 2. 20% of data were used as test data, and 80% of data were used as training data.<br>
 <br>
* Nature language processing:
 1. Each word in abstract was lemmarized and lower cased;
 2. None English words and stop words were removed;
 3. Customized stop words were removed.<br>
 <br>
* Data combination:<br>
    Cleaned abstracts in all groups were combined into single csv file (**x_train.csv** & **x_test.csv**). And their corresponding labels were also combined into single csv file with same order (**y_train.csv** & **y_test.csv**).   <br>
    <br>
* Detailed code could be found in ***data_clean.py***

#### D. Model Training
working on it

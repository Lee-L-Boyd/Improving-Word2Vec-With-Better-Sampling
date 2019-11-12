scp -i ~/1_paper/mr-key.pem mr@129.114.110.93:~/hadoop-1.2.1/1_sparse_matrix/pmi_results/$2_$1 ./pmi_results


python matrix.py $1 $2


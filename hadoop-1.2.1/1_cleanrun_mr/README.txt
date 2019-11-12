steps
1. run_word_count.sh //counts the frequency of every word in the corpus 
  // and creates a dictionary of the top 200k words 
2. run_get_contexts.sh // uses the dictionary from step one to get context
  // windows of size 5 but only for words in the dictionary
3. run_context_count_nospan.sh/run_context_count_span //aggregates the counts
  // based on either span or no span methods
4. run_row_count //gets a count of the sum of the context count for each word
5. run_pmi_nospan //uses row counts along with the context counts to compute 
  // pmi for each source-context possibility and copies the results into 
  // ../1_sparse_matrix/pmi_results
6. ../1_sparse_matrix/pmi_results makes a sparse matrix of the pmi_results
  //which can be used for the final svds step

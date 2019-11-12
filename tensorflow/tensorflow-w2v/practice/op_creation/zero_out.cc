#include "tensorflow/core/framework/op.h"
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int = 0")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
#include <deque>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
namespace {

	bool ScanWord(StringPiece* input, string* word) {
	  str_util::RemoveLeadingWhitespace(input);
	  StringPiece tmp;
	  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
	    word->assign(tmp.data(), tmp.size());
	    return true;
	  } else {
	    return false;
	  }
	}
}
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve 
    OP_REQUIRES_OK(context, context->GetAttr("preserve_index", &preserve_index_)); 
    // Check that preserve_index is positive 
    OP_REQUIRES(context, preserve_index_ >= 0, errors::InvalidArgument("Need preserve_index >= 0, got ", preserve_index_)); } 
  

  void Compute(OpKernelContext* context) override {
    string data = "this is a test is";
    StringPiece input2 = data;
    //reverseWords(&input2);
    string w;  
    int window_size = 2;
    int deque_size = 0;
    std::deque<string> previous_words;
    typedef std::unordered_map<string, int32> source_vector;
    std::unordered_map<string, source_vector> word_freq; 
    std::unordered_map<string, int32> word_idf_total; 
    while(ScanWord(&input2, &w)){
      //for(auto it = word_freq.begin(); it!=word_freq.end(); ++it){
        for (const auto& context_word: previous_words){
          //std::cout<<context_word;
          word_freq[w][context_word]+=1;
	  word_freq[context_word][w]+=1;
        }
      //}
      deque_size++;
      previous_words.push_back(w);
      //std::cout<<w;
      if (deque_size > window_size){
        //std::cout<<previous_words.front();
	previous_words.pop_front();
        deque_size--;
      }
    }
    for(const auto& it:word_freq){
      std::cout<<it.first<<": ";
      for(const auto& source_context:it.second){
        std::cout<<source_context.first<<": "<<source_context.second<<", ";
        //change counts to one or zero for all vectors
        if(source_context.second > 0){
          word_idf_total[it.first] += 1;
        }
      }
      std::cout<<std::endl<<it.first<<": "<<word_idf_total[it.first];
      std::cout<<std::endl;
    }
    word_freq.clear();
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();
    OP_REQUIRES(context, preserve_index_ < input.dimension(0), errors::InvalidArgument("preserve_index out of range"));
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(preserve_index_) = input(preserve_index_);
    std::cout << "Hello World!" << std::endl;
  }
  private: int preserve_index_;
};
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_GPU), ZeroOutOp);

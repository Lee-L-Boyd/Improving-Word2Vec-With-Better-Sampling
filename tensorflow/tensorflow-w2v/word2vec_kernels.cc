/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include <deque>
namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 1000;

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

}  // end namespace

class SkipgramWord2vecOp : public OpKernel {
 public:
  explicit SkipgramWord2vecOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pos_dist", &pos_dist_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("neg_dist", &neg_dist_));
    
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

    mutex_lock l(mu_);
    example_pos_ = corpus_size_;
    label_pos_ = corpus_size_;
    label_limit_ = corpus_size_;
    sentence_index_ = kSentenceSize;
    for (int i = 0; i < kPrecalc; ++i) {
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input,
                        &precalc_examples_[j].label);
          }
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }
    ctx->set_output(0, word_);
    ctx->set_output(1, neg_freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  int min_count_ = 500;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor pos_freq_;
  Tensor neg_freq_;
  int64 corpus_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<Example> precalc_examples_;
  int precalc_index_ = 0;
  std::vector<int32> sentence_;
  int sentence_index_ = 0;
  float pos_dist_=1;
  float neg_dist_=1;
  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int64 example_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  int32 label_limit_ GUARDED_BY(mu_);

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ wraps around at the end of corpus_. For each
  // example, we randomly generate [label_pos_, label_limit) for
  // labels.
  void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (label_pos_ >= label_limit_) {
        ++total_words_processed_;
        ++sentence_index_;
        if (sentence_index_ >= kSentenceSize) {
          sentence_index_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
            if (example_pos_ >= corpus_size_) {
              ++current_epoch_;
              example_pos_ = 0;
            }
            if (subsample_ > 0) {
              int32 word_freq = pos_freq_.flat<int32>()(corpus_[example_pos_]);
              // See Eq. 5 in http://arxiv.org/abs/1310.4546
              auto x = word_freq / (subsample_ * corpus_size_);
              auto numerator = std::pow(static_cast<float>(x),.5f)+1;
              float keep_prob = numerator/x;
                  //std::sqrt(corpus_size_*subsample_/word_freq);
              if (rng_.RandFloat() > keep_prob) {
                i--;
                continue;
              }
            }
            sentence_[i] = corpus_[example_pos_];
          }
        }
        const int32 skip = 1 + rng_.Uniform(window_size_);
        label_pos_ = std::max<int32>(0, sentence_index_ - skip);
        label_limit_ =
            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
      }
      if (sentence_index_ != label_pos_) {
        break;
      }
      ++label_pos_;
    }
    *example = sentence_[sentence_index_];
    *label = sentence_[label_pos_++];
  }


  //Todo: Change init to handle to xml files instead or multiple files if you translate the xml somewhere else

  Status Init(Env* env, const string& filename) {
    
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    StringPiece input = data;
    string w;
    corpus_size_ = 0;
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    
    //std::unordered_map<string, double> normalized_word_idf_total; 
    //std::unordered_map<string, double> normalized_word_freq;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]);
      ++corpus_size_;
    }
    
    if (corpus_size_ < window_size_ * 10) {
      return errors::InvalidArgument("The text file ", filename,
	                             " contains too little data: ",
	                             corpus_size_, " words");
    }
    
    for (const auto& p : word_freq) {
      //normalized_word_freq[p.first]=(double)p.second/corpus_size_+.000001;
      if (p.second >= min_count_) ordered.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
      << " bytes, " << corpus_size_ << " words, " << word_freq.size()
      << " unique words, " << ordered.size()
      << " unique frequent words.";
    //word_idf_total = word_freq;

    std::sort(ordered.begin(), ordered.end(),
      [](const WordFreq& x, const WordFreq& y) {
        return x.second > y.second;
      });
    

    
    std::unordered_map<string, int32> word_id;
    int32 total_counted = 0;
    
    static const int32 kUnkId = -1;

      
      

    
    
    //UNK section (handling out of dictionary)
    
    
    
    if (kUnkId != 0){
      vocab_size_ = static_cast<int32>(ordered.size()); 
    }
    else{
      vocab_size_ = static_cast<int32>(ordered.size()+1);
    }
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    Tensor pos_freq(DT_INT32, TensorShape({vocab_size_}));
    Tensor neg_freq(DT_INT32, TensorShape({vocab_size_}));
    
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      auto id = i;
      word.flat<string>()(id) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
    }
    input = data;
    if (kUnkId != 0){
      corpus_size_= total_counted;
      corpus_.reserve(corpus_size_);
      while (ScanWord(&input, &w)) {
        //Clarification: returns value using 2nd parameter as index of 1st, otherwise returns 3rd 
        auto temp_num = gtl::FindWithDefault(word_id, w, kUnkId);
        if(temp_num != kUnkId){
          corpus_.push_back(temp_num);
        }
      }
    }
    else{
      for (std::size_t i = ordered.size(); i > 0; --i) {
        freq.flat<int32>()(i) = freq.flat<int32>()(i-1);
      }
      word.flat<string>()(kUnkId) = "UNK";
      corpus_.reserve(corpus_size_);
      freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
      while (ScanWord(&input, &w)) {
        corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
      }
    }
    word_ = word;
 
    int check_cs = 0;
    int idf_total = 0;

    int window_size = window_size_;
    int deque_size = 0;
    int building_counter = 0;

    

    typedef std::unordered_map<int32, int32> source_vector;
    std::unordered_map<int32, source_vector> word_freq2;
    std::deque<int32> previous_words;
    std::unordered_map<int32, int32> word_idf_total; 

    for(const auto& w: corpus_){
      ++check_cs;
      for (const auto& context_word: previous_words){
	word_freq2[w][context_word]+=1;
	word_freq2[context_word][w]+=1;
      }
      previous_words.push_back(w);
      deque_size++;
      if (deque_size > window_size){
	previous_words.pop_front();
	deque_size--;
      }
    }
    for(const auto& it:word_freq2){
      for(const auto& source_context:it.second){
	//change counts to one or zero for all vectors
	if(source_context.second > 0){
          word_idf_total[it.first]+=1;
	  //idf_freq.flat<int32>()(it.first) += 1;
          idf_total+=1;
	}
      }
    }
    for (const auto& it:word_idf_total) {
        //normalizes it to match unigram totals
        auto idf=static_cast<int32>(it.second/(double)idf_total*corpus_size_);
        //NOTE:remove this conditional (keep long line only), just for testing pos sampling using a different distribution
        neg_freq.flat<int32>()(it.first)=static_cast<int32>((1-neg_dist_)*idf+neg_dist_*freq.flat<int32>()(it.first));
        pos_freq.flat<int32>()(it.first)=static_cast<int32>((1-pos_dist_)*idf+pos_dist_*freq.flat<int32>()(it.first));

      //auto idf_count = static_cast<int32>(1);
      //idf_freq.flat<int32>()(1) = idf_count;
    }
    word_freq2.clear();
    pos_freq_ = pos_freq;
    neg_freq_ = neg_freq;
    
    if (check_cs == corpus_size_){
      std::cout<<"count was same in idf count corpus count in idf was "<< check_cs<<std::endl;
    }
    else{
      std::cout<<"count was not same in idf count or you did not choose idf option corpus count in idf was "<< check_cs<<std::endl;
    }
    if (idf_total < corpus_size_*corpus_size_){
      std::cout<<"idf_total check passed with idf of " << idf_total << std::endl;
    }
    else{
      std::cout<<"idf_total check not passed or no idf option chosen with idf of " << idf_total << std::endl;
    }
    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);
//Todo: Change negative sampling to use idf
class NegTrainWord2vecOp : public OpKernel {
 public:
  explicit NegTrainWord2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> neg_freq;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("neg_freq", &neg_freq));
    std::vector<float> vocab_weights;
    vocab_weights.reserve(neg_freq.size());
    
    for (const auto& f : neg_freq) {
        auto r = std::pow(static_cast<float>(f), 0.75f);
	vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainWord2vecOp() { delete sampler_; }

  void Compute(OpKernelContext* ctx) override {
    
    Tensor w_in = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                errors::InvalidArgument("Must be a matrix"));
    Tensor w_out = ctx->mutable_input(1, false);
    OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                errors::InvalidArgument("w_in.shape == w_out.shape"));
    const Tensor& examples = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                errors::InvalidArgument("Must be a vector"));
    const Tensor& labels = ctx->input(3);
    OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                errors::InvalidArgument("examples.shape == labels.shape"));
    const Tensor& learning_rate = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                errors::InvalidArgument("Must be a scalar"));

    auto Tw_in = w_in.matrix<float>();
    auto Tw_out = w_out.matrix<float>();
    auto Texamples = examples.flat<int32>();
    auto Tlabels = labels.flat<int32>();
    auto lr = learning_rate.scalar<float>()();
    const int64 vocab_size = w_in.dim_size(0);
    const int64 dims = w_in.dim_size(1);
    const int64 batch_size = examples.dim_size(0);
    OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                        " vs. ", sampler_->num()));

    // Gradient accumulator for v_in.
    Tensor buf(DT_FLOAT, TensorShape({dims}));
    auto Tbuf = buf.flat<float>();

    // Scalar buffer to hold sigmoid(+/- dot).
    Tensor g_buf(DT_FLOAT, TensorShape({}));
    auto g = g_buf.scalar<float>();

    // The following loop needs 2 random 32-bit values per negative
    // sample.  We reserve 8 values per sample just in case the
    // underlying implementation changes.
    auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
    random::SimplePhilox srnd(&rnd);

    for (int64 i = 0; i < batch_size; ++i) {
      const int32 example = Texamples(i);
      DCHECK(0 <= example && example < vocab_size) << example;
      const int32 label = Tlabels(i);
      DCHECK(0 <= label && label < vocab_size) << label;
      auto v_in = Tw_in.chip<0>(example);

      // Positive: example predicts label.
      //   forward: x = v_in' * v_out
      //            l = log(sigmoid(x))
      //   backward: dl/dx = g = sigmoid(-x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      {
        auto v_out = Tw_out.chip<0>(label);
        auto dot = (v_in * v_out).sum();
        g = (dot.exp() + 1.f).inverse();
        Tbuf = v_out * (g() * lr);
        v_out += v_in * (g() * lr);
      }

      // Negative samples:
      //   forward: x = v_in' * v_sample
      //            l = log(sigmoid(-x))
      //   backward: dl/dx = g = -sigmoid(x)
      //             dl/d(v_in) = g * v_out'
      //             dl/d(v_out) = v_in' * g
      for (int j = 0; j < num_samples_; ++j) {
        const int sample = sampler_->Sample(&srnd);
        if (sample == label) continue;  // Skip.
        auto v_sample = Tw_out.chip<0>(sample);
        auto dot = (v_in * v_sample).sum();
        g = -((-dot).exp() + 1.f).inverse();
        Tbuf += v_sample * (g() * lr);
        v_sample += v_in * (g() * lr);
      }

      // Applies the gradient on v_in.
      v_in += Tbuf;
    }
  }

 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainWord2vec").Device(DEVICE_CPU), NegTrainWord2vecOp);

}  // end namespace tensorflow

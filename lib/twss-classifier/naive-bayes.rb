require 'yaml'
require 'memoist'

class Array
  extend Memoist
  def sum
    self.inject{ |s, t| s + t }
  end 
  memoize :sum
  def product
    self.inject{ |s, t| s * t }
  end
  memoize :product
end

class NaiveBayes
  extend Memoist
  def initialize(ngram_size, pos_training_examples, neg_training_examples)
    @ngram_size = ngram_size
    @pos_training_examples = pos_training_examples # Array of positive training examples.
    @neg_training_examples = neg_training_examples
  end
  
  def train(len=0)
    # Get hashes from ngrams to their (smoothed) counts.

    @pcounts = get_ngram_counts(@pos_training_examples)
    @ncounts = get_ngram_counts(@neg_training_examples)
    
    @pos_counts = @pcounts[0]
    @neg_counts = @ncounts[0]
    @pos_countsl = @pcounts[1]
    @neg_countsl = @ncounts[1]


    pos_total_count = @pos_counts.values.sum
    neg_total_count = @neg_counts.values.sum
    poslen_total = @pos_countsl.values.sum
    neglen_total = @neg_countsl.values.sum
    # Get the proportions of ngrams in each corpus.
    @probs = {} # Hash.new { |h, k| h[k] = [0.5, 0.5] }
    tprobs = {}
    (@pos_counts.keys + @neg_counts.keys).uniq.each do |ngram|
      pos_p = @pos_counts[ngram].to_f / pos_total_count *(len==1? @pos_countsl[ngram].to_f / poslen_total : 1.0)
      neg_p = @neg_counts[ngram].to_f / neg_total_count *(len==1? @neg_countsl[ngram].to_f / neglen_total : 1.0)
      tprobs[ngram] = [pos_p,neg_p]
    
    @probs = tprobs
    end
  end
  
  def test(threshold, pos_test_examples, neg_test_examples)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    pos_test_examples.each do |line|
      if self.classify(line) > threshold
        tp += 1
      else
        fn += 1
      end
    end
    
    neg_test_examples.each do |line|
      if self.classify(line) < threshold
        tn += 1
      else
        fp += 1
      end
    end

    return Hash['tp',tp, 'fp',fp,'tn', tn,'fn', fn,'precision',(1.0*tp)/(tp+fp),'recall',(tp*1.0)/(tp+fn)]   
  end
  
  # Returns the probability that `sentence` is a TWSS sentence.
  def classify(sentence)
    probs = to_ngrams(sentence, @ngram_size).map{ |word| @probs[word] || [0.5, 0.5] }
    pos_probs = probs.map{ |x| x[0] }.product
    neg_probs = probs.map{ |x| x[1] }.product
    pos_p = pos_probs / (pos_probs + neg_probs)
    return pos_p
  end
  
  def yamlize(filename)
    File.open(filename, "w") do |f|
      f.puts self.to_yaml
    end
  end
  
  def self.load_yaml(filename)
    return YAML::load(File.read(filename))
  end  

  def puts_best_pos_predictors
    total_pos_count = @pos_counts.values.sum
    @probs.to_a.sort_by{ |k, v| p = v[0] / v.sum; p }.select{ |k, v| @pos_counts[k] > 10 }.reverse.first(500).each do |k, v|
      puts [k.ljust(20), v[0] / v.sum, @pos_counts[k] ].join("\t") #if @pos_counts[k] > 10
#      puts [k.ljust(20), v[0] / v.sum ].join("\t") #if @pos_counts[k] > 10
    end
  end
  
  def puts_best_neg_predictors
    total_pos_count = @pos_counts.values.sum
    @probs.to_a.sort_by{ |k, v| p = v[1] / v.sum; p }.select{ |k, v| @pos_counts[k] > 10 }.reverse.first(500).each do |k, v|
      puts [k.ljust(20), v[1] / v.sum, @pos_counts[k] ].join("\t") #if @pos_counts[k] > 10
#      puts [k.ljust(20), v[0] / v.sum ].join("\t") #if @pos_counts[k] > 10
    end
  end
  
  #private
  
  def get_ngram_counts(sentences)
    h = Hash.new{ |h, k| h[k] = 1 } # Add-one smoothing.
    l = Hash.new{ |j,l| j[l] = 1}
    sentences.each do |sentence|
      to_ngrams(sentence, @ngram_size).each do |ngram|
        h[ngram] += 1
        l[ngram] += (1.0*ngram.length)/sentence.length     
        end
    end
    return h,l    
  end
  memoize :get_ngram_counts
  def to_ngrams(str, n)
    normalize(str, n).split.each_slice(n).to_a.map{ |x| x.join(" ") }
  end
  memoize :to_ngrams
  def normalize(str, n)
    ret = str.downcase
    ret = ret.gsub(/[^a-z0-9\s]/, "")
    # Add contextual features if we aren't dealing with unigrams.
    ret = "START " + ret + " END" if n > 1 
    ret = ret.gsub(/\s+/, " ")
    return ret
  end
  memoize :normalize
end

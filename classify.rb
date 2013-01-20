require File.expand_path('../lib/twss-classifier/naive-bayes', __FILE__)
require 'parallel'

ALL_POS_EXAMPLES = IO.readlines("data/twss-stories-parsed.txt")
ALL_NEG_EXAMPLES_FMYLIFE = IO.readlines("data/fmylife-parsed.txt")
ALL_NEG_EXAMPLES_TFLN = IO.readlines("data/texts-from-last-night-parsed.txt")

POS_TRAINING_EXAMPLES = ALL_POS_EXAMPLES.first(1000)
NEG_TRAINING_EXAMPLES = ALL_NEG_EXAMPLES_FMYLIFE.first(500) + ALL_NEG_EXAMPLES_TFLN.first(500)

POS_TEST_EXAMPLES = ALL_POS_EXAMPLES.last(1000)
NEG_TEST_EXAMPLES = ALL_NEG_EXAMPLES_FMYLIFE.last(500) + ALL_NEG_EXAMPLES_TFLN.last(500)

nb = NaiveBayes.new(1, POS_TRAINING_EXAMPLES, NEG_TRAINING_EXAMPLES)
nb.yamlize("twss-classifier.yaml")
scale = 1000
thresholds = Array.new(scale) {|i| i/(1.0*scale)}

#for thres in thresholds
#  nb.train(len=0)
#  t=nb.test(thres,POS_TEST_EXAMPLES,NEG_TEST_EXAMPLES)
#  nb.train(len=1)
#  t2=nb.test(thres,POS_TEST_EXAMPLES,NEG_TEST_EXAMPLES)
#  puts  thres.to_s+" "+ t["recall"].to_s+" "+t["precision"].to_s +
#   " " +  t["recall"].to_s + " " + t["precision"].to_s
#end

a_ = Parallel.map(thresholds) do |thres|
  nb.train(len=0)
  t=nb.test(thres,POS_TEST_EXAMPLES,NEG_TEST_EXAMPLES)
  nb.train(len=1)
  t2=nb.test(thres,POS_TEST_EXAMPLES,NEG_TEST_EXAMPLES)
  puts  thres.to_s+" "+ t["recall"].to_s+" "+t["precision"].to_s +
   " " +  t["recall"].to_s + " " + t["precision"].to_s
end



exit
false_positives = 0
total_count = 0
File.readlines("data/gutenberg-fairy-tales.txt").select{ |x| !x.strip.empty? }.map(&:strip).each do |line|
  if nb.classify(line) > threshold
    puts "* #{line}"
    false_positives += 1 
  end
  total_count += 1
end
puts false_positives
puts total_count

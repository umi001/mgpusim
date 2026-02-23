package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/gpt2"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var numIterFlag = flag.Int("num-iter", 1,
	"Number of forward pass iterations to run.")
var batchSizeFlag = flag.Int("batch-size", 1,
	"Number of sequences per batch.")
var seqLenFlag = flag.Int("seq-len", 128,
	"Sequence length (number of tokens per sequence).")
var dModelFlag = flag.Int("d-model", 768,
	"Model dimension (hidden size).")
var nHeadsFlag = flag.Int("n-heads", 12,
	"Number of attention heads.")
var nLayersFlag = flag.Int("n-layers", 12,
	"Number of transformer blocks.")
var dFFFlag = flag.Int("d-ff", 3072,
	"Feed-forward inner dimension.")
var vocabSizeFlag = flag.Int("vocab-size", 50257,
	"Vocabulary size.")
var accelConfigPath = flag.String("accel-config", "",
	"Path to accel config JSON for accelerator block offloading.")

func main() {
	flag.Parse()

	r := new(runner.Runner).Init()

	benchmark := gpt2.NewBenchmark(r.Driver())
	benchmark.NumIter = *numIterFlag
	benchmark.BatchSize = *batchSizeFlag
	benchmark.SeqLen = *seqLenFlag
	benchmark.DModel = *dModelFlag
	benchmark.NHeads = *nHeadsFlag
	benchmark.NLayers = *nLayersFlag
	benchmark.DFF = *dFFFlag
	benchmark.VocabSize = *vocabSizeFlag
	benchmark.AccelConfigPath = *accelConfigPath

	r.AddBenchmark(benchmark)

	r.Run()
}

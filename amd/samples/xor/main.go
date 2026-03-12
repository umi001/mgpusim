package main

import (
	"flag"
	"math/rand"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training_benchmarks/xor"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

var accelConfigPath = flag.String("accel-config", "",
	"Path to accel_config.json for accelerator layer offloading.")

func main() {
	rand.Seed(1)

	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := xor.NewBenchmark(runner.Driver())
	benchmark.AccelConfigPath = *accelConfigPath

	runner.AddBenchmark(benchmark)

	runner.Run()
}

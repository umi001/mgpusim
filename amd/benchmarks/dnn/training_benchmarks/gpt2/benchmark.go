// Package gpt2 implements a GPT-2-scale transformer benchmark.
//
// The benchmark allocates ~163M parameters (GPT-2 Small config) and runs
// a forward-only inference pass using synthetic random token IDs.
// Gemm-heavy operations (projections, FFN) are dispatched through the
// tensor.Operator interface and can run on either GPU or accelerator.
// LayerNorm, GELU, embedding, and attention score computation use
// host-side fallbacks.
//
// Multi-GPU is supported via data parallelism: each GPU gets its own
// parameter copy and runs independent forward passes.
package gpt2

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/acceltensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// accelConfig defines the JSON structure for accelerator configuration.
type accelConfig struct {
	AccelBlocks []int `json:"accel_blocks"`
}

// transformerBlockWeights holds the weight tensors for a single block.
type transformerBlockWeights struct {
	ln1Gamma, ln1Beta tensor.Tensor
	wq, wk, wv, wo   tensor.Tensor
	bq, bk, bv, bo   tensor.Tensor
	ln2Gamma, ln2Beta tensor.Tensor
	ffnW1, ffnB1      tensor.Tensor
	ffnW2, ffnB2      tensor.Tensor
}

// gpuInstance holds per-GPU state for data-parallel forward passes.
type gpuInstance struct {
	ctx      *driver.Context
	gpuID    int
	gpuOp    tensor.Operator
	blockOps []tensor.Operator

	tokenEmbed tensor.Tensor
	posEmbed   tensor.Tensor
	blocks     []transformerBlockWeights
	finalLNGamma, finalLNBeta tensor.Tensor
	outputW tensor.Tensor
}

// Benchmark implements the GPT-2 transformer benchmark.
type Benchmark struct {
	driver *driver.Driver
	ctx    *driver.Context
	gpus   []int

	instances []*gpuInstance

	// Model config — GPT-2 Small defaults.
	DModel    int
	NHeads    int
	NLayers   int
	DFF       int
	VocabSize int
	SeqLen    int
	BatchSize int
	NumIter   int

	accelBlockSet   map[int]bool
	AccelConfigPath string
}

// NewBenchmark creates a new GPT-2 benchmark with default config.
func NewBenchmark(d *driver.Driver) *Benchmark {
	b := &Benchmark{
		driver:    d,
		ctx:       d.Init(),
		DModel:    768,
		NHeads:    12,
		NLayers:   12,
		DFF:       3072,
		VocabSize: 50257,
		SeqLen:    1024,
		BatchSize: 1,
		NumIter:   1,
	}

	return b
}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	b.gpus = gpuIDs
}

// Run executes the benchmark. //nolint:funlen
func (b *Benchmark) Run() {
	b.accelBlockSet = b.loadAccelConfig()

	for _, gpuID := range b.gpus {
		inst := b.createGPUInstance(gpuID)
		b.instances = append(b.instances, inst)
	}

	b.reportParamCount(b.instances[0])

	for iter := 0; iter < b.NumIter; iter++ {
		for i, inst := range b.instances {
			tokens := b.randomTokens()
			logits := b.forward(inst, tokens)
			log.Printf("Iteration %d, GPU %d (id=%d): logits shape %v",
				iter, i, inst.gpuID, logits.Size())
			inst.gpuOp.Free(logits)
		}
	}
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}

func (b *Benchmark) createGPUInstance( //nolint:funlen
	gpuID int,
) *gpuInstance {
	ctx := b.driver.InitWithExistingPID(b.ctx)
	b.driver.SelectGPU(ctx, gpuID)

	gpuOp := gputensor.NewGPUOperator(b.driver, ctx)

	var accelOp tensor.Operator

	numAccel := b.driver.GetNumAccelerators()
	if numAccel > 0 && len(b.accelBlockSet) > 0 {
		accelIdx := len(b.instances) % numAccel
		b.driver.SelectAccelerator(ctx, accelIdx)
		accelOp = acceltensor.NewOperator(b.driver, ctx)
	} else {
		accelOp = gpuOp
	}

	blockOps := make([]tensor.Operator, b.NLayers)
	for i := 0; i < b.NLayers; i++ {
		if b.accelBlockSet[i] {
			blockOps[i] = accelOp
		} else {
			blockOps[i] = gpuOp
		}
	}

	inst := &gpuInstance{
		ctx:      ctx,
		gpuID:    gpuID,
		gpuOp:    gpuOp,
		blockOps: blockOps,
	}

	b.allocateParameters(inst)
	b.randomizeParameters(inst)

	return inst
}

func (b *Benchmark) loadAccelConfig() map[int]bool {
	set := make(map[int]bool)

	path := b.AccelConfigPath
	if path == "" {
		return set
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return set
	}

	var cfg accelConfig

	err = json.Unmarshal(data, &cfg)
	if err != nil {
		log.Printf("Warning: invalid accel config: %v", err)
		return set
	}

	for _, id := range cfg.AccelBlocks {
		set[id] = true
	}

	return set
}

func (b *Benchmark) allocateParameters(inst *gpuInstance) {
	op := inst.gpuOp

	inst.tokenEmbed = op.Create([]int{b.VocabSize * b.DModel})
	inst.posEmbed = op.Create([]int{b.SeqLen * b.DModel})

	inst.blocks = make([]transformerBlockWeights, b.NLayers)
	for i := 0; i < b.NLayers; i++ {
		blk := &inst.blocks[i]
		blk.ln1Gamma = op.Create([]int{b.DModel})
		blk.ln1Beta = op.Create([]int{b.DModel})
		blk.wq = op.Create([]int{b.DModel * b.DModel})
		blk.wk = op.Create([]int{b.DModel * b.DModel})
		blk.wv = op.Create([]int{b.DModel * b.DModel})
		blk.wo = op.Create([]int{b.DModel * b.DModel})
		blk.bq = op.Create([]int{b.DModel})
		blk.bk = op.Create([]int{b.DModel})
		blk.bv = op.Create([]int{b.DModel})
		blk.bo = op.Create([]int{b.DModel})
		blk.ln2Gamma = op.Create([]int{b.DModel})
		blk.ln2Beta = op.Create([]int{b.DModel})
		blk.ffnW1 = op.Create([]int{b.DModel * b.DFF})
		blk.ffnB1 = op.Create([]int{b.DFF})
		blk.ffnW2 = op.Create([]int{b.DFF * b.DModel})
		blk.ffnB2 = op.Create([]int{b.DModel})
	}

	inst.finalLNGamma = op.Create([]int{b.DModel})
	inst.finalLNBeta = op.Create([]int{b.DModel})
	inst.outputW = op.Create([]int{b.DModel * b.VocabSize})
}

func (b *Benchmark) randomizeParameters( //nolint:funlen
	inst *gpuInstance,
) {
	op := inst.gpuOp
	initParam := func(t tensor.Tensor, fanIn int) {
		n := t.NumElement()
		data := make([]float64, n)
		scale := 1.0 / math.Sqrt(float64(fanIn))
		for i := range data {
			data[i] = (rand.Float64() - 0.5) * 2 * scale
		}
		op.Init(t, data)
	}

	onesParam := func(t tensor.Tensor) {
		n := t.NumElement()
		data := make([]float64, n)
		for i := range data {
			data[i] = 1.0
		}
		op.Init(t, data)
	}

	zerosParam := func(t tensor.Tensor) {
		op.Clear(t)
	}

	initParam(inst.tokenEmbed, b.VocabSize)
	initParam(inst.posEmbed, b.SeqLen)

	for i := 0; i < b.NLayers; i++ {
		blk := &inst.blocks[i]
		onesParam(blk.ln1Gamma)
		zerosParam(blk.ln1Beta)
		initParam(blk.wq, b.DModel)
		initParam(blk.wk, b.DModel)
		initParam(blk.wv, b.DModel)
		initParam(blk.wo, b.DModel)
		zerosParam(blk.bq)
		zerosParam(blk.bk)
		zerosParam(blk.bv)
		zerosParam(blk.bo)
		onesParam(blk.ln2Gamma)
		zerosParam(blk.ln2Beta)
		initParam(blk.ffnW1, b.DModel)
		zerosParam(blk.ffnB1)
		initParam(blk.ffnW2, b.DFF)
		zerosParam(blk.ffnB2)
	}

	onesParam(inst.finalLNGamma)
	zerosParam(inst.finalLNBeta)
	initParam(inst.outputW, b.DModel)
}

func (b *Benchmark) randomTokens() []float64 {
	tokens := make([]float64, b.BatchSize*b.SeqLen)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(b.VocabSize))
	}

	return tokens
}

// forward runs the full GPT-2 forward pass and returns logits.
func (b *Benchmark) forward(
	inst *gpuInstance, tokens []float64,
) tensor.Tensor {
	batchSeq := b.BatchSize * b.SeqLen

	// Embedding: token embed + position embed (host fallback).
	x := b.embed(inst, tokens)

	// Transformer blocks.
	for i := 0; i < b.NLayers; i++ {
		x = b.transformerBlock(inst, i, x, batchSeq)
	}

	// Final layer norm.
	x = b.layerNormReplace(
		inst.gpuOp, x, inst.finalLNGamma, inst.finalLNBeta)

	// Output projection: logits = x @ outputW.
	logits := b.linear(inst.gpuOp, x, inst.outputW,
		nil, b.DModel, b.VocabSize)
	inst.gpuOp.Free(x)

	return logits
}

// transformerBlock runs a single transformer block with pre-norm residuals.
func (b *Benchmark) transformerBlock(
	inst *gpuInstance, blockIdx int,
	x tensor.Tensor, batchSeq int,
) tensor.Tensor {
	op := inst.blockOps[blockIdx]
	blk := &inst.blocks[blockIdx]

	// Self-attention with residual.
	residual := op.Clone(x)
	normed := b.layerNormReplace(op, x, blk.ln1Gamma, blk.ln1Beta)
	attnOut := b.multiHeadAttention(inst, op, normed, blk, batchSeq)
	x = op.ScaleAdd(1, 1, residual, attnOut)
	op.Free(residual)
	op.Free(attnOut)

	// FFN with residual.
	residual2 := op.Clone(x)
	normed2 := b.layerNormReplace(op, x, blk.ln2Gamma, blk.ln2Beta)
	ffnOut := b.ffn(op, normed2, blk)
	x = op.ScaleAdd(1, 1, residual2, ffnOut)
	op.Free(residual2)
	op.Free(ffnOut)

	return x
}

// embed performs token + position embedding via host fallback.
func (b *Benchmark) embed(
	inst *gpuInstance, tokens []float64,
) tensor.Tensor {
	op := inst.gpuOp
	batchSeq := b.BatchSize * b.SeqLen

	tokEmbData := inst.tokenEmbed.Vector()
	posEmbData := inst.posEmbed.Vector()

	outData := make([]float64, batchSeq*b.DModel)

	for i := 0; i < batchSeq; i++ {
		tokenID := int(tokens[i])
		posID := i % b.SeqLen
		tokOffset := tokenID * b.DModel
		posOffset := posID * b.DModel

		for j := 0; j < b.DModel; j++ {
			outData[i*b.DModel+j] =
				tokEmbData[tokOffset+j] + posEmbData[posOffset+j]
		}
	}

	out := op.Create([]int{batchSeq, b.DModel})
	op.Init(out, outData)

	return out
}

// layerNormReplace applies layer norm via host fallback and frees the
// input tensor, returning the normalized output.
func (b *Benchmark) layerNormReplace(
	op tensor.Operator,
	x, gamma, beta tensor.Tensor,
) tensor.Tensor {
	out := b.layerNorm(op, x, gamma, beta)
	op.Free(x)

	return out
}

// layerNorm applies layer norm via host fallback.
func (b *Benchmark) layerNorm(
	op tensor.Operator,
	x, gamma, beta tensor.Tensor,
) tensor.Tensor {
	xData := x.Vector()
	gammaData := gamma.Vector()
	betaData := beta.Vector()

	size := x.Size()
	rows := size[0]
	cols := size[1]
	outData := make([]float64, rows*cols)

	const epsilon = 1e-5

	for i := 0; i < rows; i++ {
		offset := i * cols

		mean := 0.0
		for j := 0; j < cols; j++ {
			mean += xData[offset+j]
		}
		mean /= float64(cols)

		variance := 0.0
		for j := 0; j < cols; j++ {
			d := xData[offset+j] - mean
			variance += d * d
		}
		variance /= float64(cols)

		invStd := 1.0 / math.Sqrt(variance+epsilon)

		for j := 0; j < cols; j++ {
			normalized := (xData[offset+j] - mean) * invStd
			outData[offset+j] = gammaData[j]*normalized + betaData[j]
		}
	}

	out := op.Create(size)
	op.Init(out, outData)

	return out
}

// linear performs output = input @ weight + bias via Gemm.
func (b *Benchmark) linear(
	op tensor.Operator,
	input, weight, bias tensor.Tensor,
	inputDim, outputDim int,
) tensor.Tensor {
	batchSeq := input.Size()[0]

	inMat := op.Reshape(input, []int{batchSeq, inputDim})
	wMat := op.Reshape(weight, []int{inputDim, outputDim})

	var cMat tensor.Tensor
	if bias != nil {
		bRepeated := op.Repeat(bias, batchSeq)
		cMat = op.Reshape(bRepeated, []int{batchSeq, outputDim})
		op.Free(bRepeated)
	} else {
		cMat = op.Zeros([]int{batchSeq, outputDim})
	}

	betaVal := 0.0
	if bias != nil {
		betaVal = 1.0
	}

	out := op.Gemm(false, false, 1, betaVal, inMat, wMat, cMat)

	op.Free(inMat)
	op.Free(wMat)
	op.Free(cMat)

	return out
}

// multiHeadAttention computes multi-head self-attention via Gemm
// projections and host-fallback attention scores.
func (b *Benchmark) multiHeadAttention( //nolint:funlen
	inst *gpuInstance,
	op tensor.Operator,
	x tensor.Tensor,
	blk *transformerBlockWeights,
	batchSeq int,
) tensor.Tensor {
	headDim := b.DModel / b.NHeads

	// Q, K, V projections via Gemm.
	q := b.linear(op, x, blk.wq, blk.bq, b.DModel, b.DModel)
	k := b.linear(op, x, blk.wk, blk.bk, b.DModel, b.DModel)
	v := b.linear(op, x, blk.wv, blk.bv, b.DModel, b.DModel)
	op.Free(x)

	// Attention scores on CPU.
	qData := q.Vector()
	kData := k.Vector()
	vData := v.Vector()
	op.Free(q)
	op.Free(k)
	op.Free(v)

	attnOutData := make([]float64, batchSeq*b.DModel)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for batch := 0; batch < b.BatchSize; batch++ {
		for h := 0; h < b.NHeads; h++ {
			b.attentionHead(
				qData, kData, vData, attnOutData,
				batch, h, headDim, scale,
			)
		}
	}

	attnOut := op.Create([]int{batchSeq, b.DModel})
	op.Init(attnOut, attnOutData)

	// Output projection via Gemm.
	projected := b.linear(op, attnOut, blk.wo, blk.bo, b.DModel, b.DModel)
	op.Free(attnOut)

	return projected
}

// attentionHead computes attention for a single head of a single batch.
func (b *Benchmark) attentionHead(
	qData, kData, vData, outData []float64,
	batch, head, headDim int,
	scale float64,
) {
	seq := b.SeqLen

	// Compute scores = Q @ K^T / sqrt(d_k), apply softmax, then @ V.
	scores := make([]float64, seq*seq)

	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			dot := 0.0
			for d := 0; d < headDim; d++ {
				qi := b.mhaIndex(batch, i, head, d)
				ki := b.mhaIndex(batch, j, head, d)
				dot += qData[qi] * kData[ki]
			}

			scores[i*seq+j] = dot * scale
		}
	}

	// Softmax per row.
	for i := 0; i < seq; i++ {
		maxVal := math.Inf(-1)
		for j := 0; j < seq; j++ {
			if scores[i*seq+j] > maxVal {
				maxVal = scores[i*seq+j]
			}
		}

		sum := 0.0
		for j := 0; j < seq; j++ {
			scores[i*seq+j] = math.Exp(scores[i*seq+j] - maxVal)
			sum += scores[i*seq+j]
		}

		for j := 0; j < seq; j++ {
			scores[i*seq+j] /= sum
		}
	}

	// Output = scores @ V.
	for i := 0; i < seq; i++ {
		for d := 0; d < headDim; d++ {
			val := 0.0
			for j := 0; j < seq; j++ {
				vi := b.mhaIndex(batch, j, head, d)
				val += scores[i*seq+j] * vData[vi]
			}

			outIdx := b.mhaIndex(batch, i, head, d)
			outData[outIdx] = val
		}
	}
}

// mhaIndex computes the flat index for multi-head attention data stored
// as (batch*seq, d_model) = (batch*seq, n_heads*head_dim).
func (b *Benchmark) mhaIndex(batch, seqPos, head, d int) int {
	headDim := b.DModel / b.NHeads
	return (batch*b.SeqLen+seqPos)*b.DModel + head*headDim + d
}

// ffn applies the feed-forward network: GELU(x @ W1 + b1) @ W2 + b2.
func (b *Benchmark) ffn(
	op tensor.Operator,
	x tensor.Tensor,
	blk *transformerBlockWeights,
) tensor.Tensor {
	// First linear: x @ W1 + b1.
	h := b.linear(op, x, blk.ffnW1, blk.ffnB1, b.DModel, b.DFF)
	op.Free(x)

	// GELU activation (host fallback).
	h = b.geluReplace(op, h)

	// Second linear: h @ W2 + b2.
	out := b.linear(op, h, blk.ffnW2, blk.ffnB2, b.DFF, b.DModel)
	op.Free(h)

	return out
}

// geluReplace applies GELU activation via host fallback and frees input.
func (b *Benchmark) geluReplace(
	op tensor.Operator,
	x tensor.Tensor,
) tensor.Tensor {
	data := x.Vector()
	outData := make([]float64, len(data))

	for i, v := range data {
		outData[i] = 0.5 * v * (1.0 + math.Tanh(
			math.Sqrt(2.0/math.Pi)*(v+0.044715*v*v*v)))
	}

	out := op.Create(x.Size())
	op.Init(out, outData)
	op.Free(x)

	return out
}

func (b *Benchmark) reportParamCount(inst *gpuInstance) {
	total := inst.tokenEmbed.NumElement() + inst.posEmbed.NumElement()

	for i := 0; i < b.NLayers; i++ {
		blk := &inst.blocks[i]
		total += blk.ln1Gamma.NumElement() + blk.ln1Beta.NumElement()
		total += blk.wq.NumElement() + blk.wk.NumElement()
		total += blk.wv.NumElement() + blk.wo.NumElement()
		total += blk.bq.NumElement() + blk.bk.NumElement()
		total += blk.bv.NumElement() + blk.bo.NumElement()
		total += blk.ln2Gamma.NumElement() + blk.ln2Beta.NumElement()
		total += blk.ffnW1.NumElement() + blk.ffnB1.NumElement()
		total += blk.ffnW2.NumElement() + blk.ffnB2.NumElement()
	}

	total += inst.finalLNGamma.NumElement() + inst.finalLNBeta.NumElement()
	total += inst.outputW.NumElement()

	log.Printf("GPT-2 parameter count: %s (%d)",
		formatParams(total), total)
}

func formatParams(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	default:
		return fmt.Sprintf("%d", n)
	}
}

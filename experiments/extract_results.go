// extract_results.go — Extracts interconnect comparison metrics from all experiments.
// Usage: go run experiments/extract_results.go
package main

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

type result struct {
	benchmark string
	config    string
	bw        string
	latency   string
	// Accel metrics (sum of both accelerators)
	busyTime      float64
	opCount       float64
	computeCycles float64
	memReqs       float64
	readBytes     float64
	writeBytes    float64
	// Overall
	kernelTime float64
}

func main() {
	resultsDir := "experiments/results"
	if len(os.Args) > 1 {
		resultsDir = os.Args[1]
	}

	var results []result

	benchmarks := []string{"minerva", "lenet", "vgg16"}
	configs := []string{
		"pcie_gen4_x16",
		"pcie_gen5_x16",
		"pcie_gen6_x16",
		"elec_high_bw",
		"optical_low",
		"optical_mid",
		"optical_high",
	}
	configLabels := map[string]string{
		"pcie_gen4_x16": "PCIe4 (32GB/s, 50cy)",
		"pcie_gen5_x16": "PCIe5 (64GB/s, 40cy)",
		"pcie_gen6_x16": "PCIe6 (128GB/s, 30cy)",
		"elec_high_bw":  "Elec (256GB/s, 20cy)",
		"optical_low":   "Opt-Lo (512GB/s, 5cy)",
		"optical_mid":   "Opt-Mid (1TB/s, 3cy)",
		"optical_high":  "Opt-Hi (2TB/s, 2cy)",
	}

	for _, bench := range benchmarks {
		for _, cfg := range configs {
			dbPath := filepath.Join(resultsDir, bench, cfg, "metrics.sqlite3")
			if _, err := os.Stat(dbPath); err != nil {
				continue
			}
			r := extractMetrics(dbPath, bench, cfg, configLabels[cfg])
			results = append(results, r)
		}
	}

	if len(results) == 0 {
		fmt.Println("No results found in", resultsDir)
		return
	}

	// Print summary table
	fmt.Println("=" + strings.Repeat("=", 139))
	fmt.Printf("%-10s %-28s %12s %8s %12s %14s %14s %12s\n",
		"Benchmark", "Interconnect", "Kernel(ms)", "AccOps", "AccBusy(us)",
		"ReadMB", "WriteMB", "MemReqs")
	fmt.Println(strings.Repeat("-", 140))

	sort.Slice(results, func(i, j int) bool {
		if results[i].benchmark != results[j].benchmark {
			return results[i].benchmark < results[j].benchmark
		}
		return configOrder(results[i].config) < configOrder(results[j].config)
	})

	prevBench := ""
	for _, r := range results {
		if r.benchmark != prevBench && prevBench != "" {
			fmt.Println(strings.Repeat("-", 140))
		}
		prevBench = r.benchmark
		fmt.Printf("%-10s %-28s %12.3f %8.0f %12.1f %14.2f %14.2f %12.0f\n",
			r.benchmark, r.latency,
			r.kernelTime*1000,
			r.opCount,
			r.busyTime*1e6,
			r.readBytes/1e6,
			r.writeBytes/1e6,
			r.memReqs)
	}
	fmt.Println("=" + strings.Repeat("=", 139))

	// Speedup analysis
	fmt.Println("\nSpeedup vs PCIe Gen4 (by accel busy_time):")
	fmt.Printf("%-10s", "Benchmark")
	for _, cfg := range configs {
		fmt.Printf(" %12s", cfg[:min(12, len(cfg))])
	}
	fmt.Println()

	for _, bench := range benchmarks {
		fmt.Printf("%-10s", bench)
		var baseline float64
		for _, cfg := range configs {
			for _, r := range results {
				if r.benchmark == bench && r.config == cfg {
					if cfg == "pcie_gen4_x16" {
						baseline = r.busyTime
					}
					if baseline > 0 {
						fmt.Printf(" %11.2fx", baseline/r.busyTime)
					} else {
						fmt.Printf(" %12s", "N/A")
					}
				}
			}
		}
		fmt.Println()
	}
}

func extractMetrics(dbPath, bench, config, label string) result {
	r := result{benchmark: bench, config: config, latency: label}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return r
	}
	defer db.Close()

	rows, err := db.Query("SELECT Location, What, Value FROM mgpusim_metrics")
	if err != nil {
		return r
	}
	defer rows.Close()

	for rows.Next() {
		var loc, what string
		var val float64
		rows.Scan(&loc, &what, &val)

		if strings.Contains(loc, "Accel") && strings.Contains(loc, "AccelUnit") {
			switch what {
			case "busy_time":
				r.busyTime += val
			case "op_count":
				r.opCount += val
			case "total_compute_cycles":
				r.computeCycles += val
			case "total_mem_reqs":
				r.memReqs += val
			case "total_read_bytes":
				r.readBytes += val
			case "total_write_bytes":
				r.writeBytes += val
			}
		}
		if loc == "Driver" && what == "kernel_time" {
			r.kernelTime = val
		}
	}
	return r
}

func configOrder(cfg string) int {
	order := map[string]int{
		"pcie_gen4_x16": 0,
		"pcie_gen5_x16": 1,
		"pcie_gen6_x16": 2,
		"elec_high_bw":  3,
		"optical_low":   4,
		"optical_mid":   5,
		"optical_high":  6,
	}
	if v, ok := order[cfg]; ok {
		return v
	}
	return 99
}

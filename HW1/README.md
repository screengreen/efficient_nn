Sukhov Andrei
# HW1 - analytical cost model for SmallCNN 

Predict inference latency, energy and peak memory of SmallCNN (models.py) from analytical
FLOP / byte counts, calibrate the constants on real measurements, and check how far off they are.

## Hardware and software

Measurements were taken on Google Colab, NVIDIA Tesla T4 16 GB.
cudnn.benchmark = False, cudnn.allow_tf32 = False,
matmul.allow_tf32 = False, 

## How to reproduce

1. Open measure.py in Colab as a notebook, pick a T4 GPU runtime, run it top to bottom.
   It warms the GPU up for 10 s on a heavy shape first so the clocks settle at their sustained
   level, then sweeps all (S, B) points back to back with no cooldown. Each point: 5 warmup
   passes, median of 20 timed passes with torch.cuda.synchronize() around each, peak memory
   from max_memory_allocated(), and energy from the NVML total-energy counter averaged over
   enough repeats to cover at least 0.5 s (the counter ticks in mJ, short regions would be pure noise).
   Takes about 15 min and writes measurements.csv.
2. Put it in results/measurements.csv.
3. python calibrate.py - fits both models, prints metrics, writes results/theta.json.
4. python plot.py - writes the four figures into results/figures/.

The grid is S in {32,64,128,224,256,384,512,768,1024} times B in {1,2,4,…,512} plus 4 random
extra sizes {48,112,448,496} and 3 random extra batches {65,70,78} (random.seed(42)).
A point is validation iff its S or B is one of the extras - so the model is never fitted
on those shapes. 169 points total, 162 measured (84 train / 78 val), 7 not run.

## The equations

flops(S, B) sums per layer. Convs are 2·k²·Cin·Cout·S_out²·B, linear 2·B·in·out,
plus ReLU / maxpool / pooling terms. The factor 2 is multiply + add.

bytes_moved(S, B) is a traffic estimate: for every layer, input activations + weights +
output activations. ReLU is skipped since it is in-place. This assumes nothing stays in
cache between layers, so it is an upper bound on real DRAM traffic.

memory(S, B) is params plus the largest (input + output) activation pair over all layers,
i.e. a peak-footprint estimate assuming activations are freed as soon as they are consumed.

latency(S, B, θ) = max(θ₀, θ₁·f + θ₂′·m) with f in GFLOP, m in GB.
A launch floor, then compute time plus memory time.
Note: the third slot in theta.json is stored as θ₂ and the code uses θ₂′ = 1 − θ₂,
so the readable seconds per GB number is the seconds_per_gb field, not latency[2].

energy(S, B, θ) = max(E₀, ε₁·f·(1 + ε₂·m)) - a fixed cost per launch, then joules per
GFLOP that grow with the amount of traffic.

Both are fitted in calibrate.py with least_squares on log residuals, on train points only.
Latency spans 0.47 ms … 983 ms and energy 24 mJ … 63.6 J, so absolute residuals would have fitted
the five biggest points and ignored everything else; log residuals weight each point by its
relative error, which is what MAPE reports.

## Results

Calibrated parameters: θ₀ = 0.609 ms (launch floor); θ₁ = 0.223 ms/GFLOP (4484 GFLOP/s effective compute);
θ₂′ = 16.0 ms/GB (62.5 GB/s effective bandwidth); E₀ = 34.7 mJ (energy floor); ε₁ = 26.5 mJ/GFLOP;
ε₂ = 0.0115 /GB (traffic penalty).

Latency train: MAE 13.5 ms, MAPE 18.9 %, median APE 17.3 %, p90 APE 28.1 %.
Latency val: MAE 22.9 ms, MAPE 20.1 %, median APE 21.6 %, p90 APE 28.7 %.
Energy train: MAE 0.74 J, MAPE 16.8 %, median APE 14.5 %, p90 APE 33.4 %.
Energy val: MAE 1.05 J, MAPE 20.1 %, median APE 17.1 %, p90 APE 35.9 %.

Sanity checks on the constants: 4484 GFLOP/s is 55 % of the T4's 8.1 TFLOP/s FP32 peak, which is
believable for FP32 convs with no autotuning. The energy floor over the latency floor gives
34.7 mJ / 0.609 ms ≈ 57 W, right at the near-idle draw of the board. 

Figures in results/figures/ (latency and energy vs flops and bytes): dots are measured train,
triangles measured validation, lines are the model, one line per S with B running along it,
colour is log2(S), both axes log.

## Memory model and the 7 missing points

The rows without latency are (496,512) (512,512) (768,256) (768,512) (1024,128) (1024,256)
(1024,512). memory() predicts 5.17–22.0 GiB for them, and only (1024,512) actually exceeds
the T4's ~14.9 GiB - so the formula on its own does not explain why they failed.

Measured memory usage is roughly double the formula’s prediction, since the formula captures only parameters and main activations, missing fragments like workspace and allocator overhead; for example, the largest successful run (448,512) used 8.69 GiB versus a 4.21 GiB estimate. 

## Discussion

The models get the shape of the curves right - the floor, the knee and the slope are all in the
right place on all four plots - but both sit around 20 % validation MAPE.

The dominant source is a throughput cliff in the batch dimension. Median achieved throughput is
~2830 GFLOP/s at B=64 and 2749 at B=65, then drops to 1833 at B=70 and stays near 1870
for 128, 256 and 512. FLOPs and bytes only grow by 9 % between B=65 and B=70, so no smooth
function of (f, m) can produce a 1.5× jump in time - this is cuDNN switching algorithm or tile
shape, and neither flops nor bytes_moved carries that information. It is visible in the plots
as the measured points splitting into two parallel bands that the single predicted line runs
between. Most of the remaining error is this.

The second issue is that flops and bytes_moved are nearly collinear here: arithmetic intensity
is ~89 GFLOP/GB for everything except the smallest inputs, and corr(log f, log m) = 0.99. That
means the two terms cannot really be separated - at the median point the compute term is 55 % of
the predicted time and the memory term 45 %, but that split is not identifiable, and a classic
roofline max(f/P, m/BW) degenerates outright (one branch simply never wins and its coefficient
stays at whatever it was initialised to). The additive form is used because it at least keeps both
inputs alive.

Energy needed almost no separate work. Measured board power is 66.9 W median (39–82 W range,
against a 70 W TDP) and corr(energy, latency) = 0.9996, so energy is essentially power × time
with power nearly constant. The energy model is the latency model rescaled and inherits the same
error structure - same batch cliff, same ~20 % MAPE. The one place it differs is the small-input
end, where power sags toward idle (~52 W median at B=1) and the constant-power assumption is
worst; that is what the fatter p90 (36 % vs 29 %) comes from.


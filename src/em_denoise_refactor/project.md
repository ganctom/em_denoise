#em-denoise
1. Top-Down architecture (DONE)
2. Bottom-up building (DONE)
3. Validation (In-progress)


## 1. Top-down architecture definition 


**System Context & Domain:**

Design software architecture for an SBF-SEM connectomics denoising pipeline. The target dataset consists of 500k - 2 Million, 8-bit, 4096x3072 `.tif` images 
of epoxy-embedded neural tissue, hosted on a company file server. Dataset variance includes dynamic SNR, fluctuating Z-resolution, and variable tile counts per 
section. Datasets are acquired using SBEMImage software and as it is a tiled image acquisiton, there are multiple .tif images (tiles) per each slice (section number). 
The images are stored in "tiles" directory where tiles' directories are with naming convention 'tXXXX', where XXXX is tile_ID e.g. 't0009', 't0109' etc. Raw images 
filepath for an image with tileID 209, section number 13951 and within grid 0, then typically looks as follows:  

/tachyon/groups/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20260201_RoLi_F1/tiles/g0000/t0209/20260201_RoLi_F1_g0000_t0209_s13951.tif


**Objective:**

Optimize dataset fidelity for human-in-the-loop proofreading of semantic segmentations. The neural network must prioritize the preservation of high-frequency spatial 
features: axon cross-sections within bundles, synaptic vesicles, synaptic junction densities, and lipid bilayer separations.

**Methodology & Execution Environment:**

- **Architecture Phase 1 (Data Preparation):** A remote-to-local patch extraction pipeline. The system must sample representative regions of interest from the remote full-resolution tiles, extract smaller crops (patches), and stage them onto a local SSD in a high-throughput binary format (e.g., LMDB, HDF5) optimized for random access read speeds.
    
- **Architecture Phase 2 (Grid Search & Training):** A supervised learning paradigm utilizing the local patch store. The training loop must execute completely offline, reading only from the local SSD.
    
- **Data Augmentation:** Raw patches serve as ground truth (targets). Procedurally generated artificial noise must be injected on-the-fly via the data loader to create input tensors, preventing cache bloat.
    
- **Hardware:** Apple M4 Pro (48GB Unified Memory). The architecture must explicitly support PyTorch targeting the `mps` backend for high-speed hyperparameter grid searches.
    

**Deliverables:**

Output the modular decomposition as Python class stubs (`.pyi` style) with fully typed signatures and docstrings. Utilize `typing.Protocol` to define boundaries between components via dependency injection.

Do not write implementation logic. Restrict output to interface contracts for:

1. Remote data sampler, patch extractor, and local binary store builder (Phase 1 ETL).
    
2. The local patch PyTorch `Dataset` loader and synthetic noise injection pipeline.
    
3. The neural network architecture topology.
    
4. The `mps`-optimized grid-search training and inference engines.


# 2. Bottom-up building

To transition from interface contracts (`.pyi`) to a production-grade execution pipeline, implementation should follow a **bottom-up, risk-decoupled strategy**.

Rather than building top-down from the remote ETL layer—which introduces network latency, I/O bottlenecks, and external service dependencies—start by validating compute, memory management, and local storage I/O on the M4 Pro architecture.

## Phase 1: Static Type Verification & Harness Setup

Before writing business logic, configure static type checking to enforce the `.pyi` protocols during development.

1. **Directory Layout**: Mirror your stub definitions with concrete `.py` modules.
    
    ```
    src/
    ├── etl/
    │   ├── interfaces.pyi
    │   └── lmdb_store.py
    ├── data/
    │   ├── interfaces.pyi
    │   ├── dataset.py
    │   └── noise.py
    ├── models/
    │   ├── interfaces.pyi
    │   └── connectomics_net.py
    └── execution/
        ├── interfaces.pyi
        ├── mps_manager.py
        └── inference.py
    ```
    
2. **Contract Verification**: Use `pyright` or `mypy` in strict mode to ensure concrete implementations satisfy `typing.Protocol` structural subtyping without explicit inheritance.
    

## Phase 2: Implementation Sequencing Roadmap (DONE)

```
[1. Local LMDB & Dataset]  --->  [2. MPS Execution Harness]  --->  [3. Network Topology]
                                                                        |
[5. Remote ETL Pipeline]   <---  [4. Single-TIF Inference Engine]  <-----
```

### Step 1: Local Data Layer & Storage Kernel (`Data.py`)

- **Objective**: Build the local zero-copy data loader before touching remote hardware or complex neural networks.
    
- **Deliverables**:
    
    - Implement `LocalPatchStore` using `lmdb` (`read-only`, `lock=False`, `meminit=False`) for low-overhead memory-mapped I/O (`mmap`).
        
    - Implement a baseline `NoiseSynthesizer` using vectorised PyTorch operations (Poisson-Gaussian approximation + structural streak injection).
        
    - Build `SBFSEMDataset` and benchmark `DataLoader` throughput using synthetic in-memory arrays. Target $>10,000$ patches/sec.
        

### Step 2: Apple Silicon Hardware Harness (`Execution.py`)

- **Objective**: Prove `mps` backend stability and establish zero-leak memory management under high tensor throughput on 48GB Unified Memory.
    
- **Deliverables**:
    
    - Implement `MPSDeviceManager` with explicit unified memory purging routines (`torch.mps.empty_cache()`, `gc.collect()`).
        
    - Implement a dummy execution loop to benchmark memory fragmentation under high batch sizes ($N \ge 64$, $512 \times 512$ patches).
        

### Step 3: Neural Network Topology (`Architecture.py`)

- **Objective**: Implement concrete module blocks targeting high-frequency structural preservation.
    
- **Deliverables**:
    
    - Implement `FeatureExtractionEncoder` and `FeatureReconstructionDecoder` using residual blocks with depthwise separable convolutions or local spatial attention (e.g., NAFBlock / Restormer primitives).
        
    - Integrate `HighFreqAttentionBlock` (e.g., Laplacian edge-guided attention or high-pass residual gating) to bias gradient updates toward lipid bilayers and synaptic vesicles.
        

### Step 4: Standalone Image Inference Engine (`Inference.py`)

- **Objective**: Validate model output quality visually on single raw SBF-SEM `.tif` files.
    
- **Deliverables**:
    
    - Implement `InferenceEngine.process_raw_tif`.
        
    - Write a spatial windowing system (e.g., 2D sliding tile with 50% overlap and 2D Cosine/Hann window blending) using `tifffile` and `mmap` to prevent block-edge artifacts during reconstruction.
        

### Step 5: Remote ETL Pipeline (`ETL.py`)

- **Objective**: Pull unstitched tiles from remote storage and stage them locally.
    
- **Deliverables**:
    
    - Implement `RemoteStorageClient` using asynchronous HTTP (`aiohttp`) or high-throughput S3 SDK (`boto3` threadpools).
        
    - Implement `ROISampler` and `PatchExtractor` to parse raw section-to-tile maps, cropping sub-volumes directly into the `BinaryStoreWriter` (LMDB).
-

# Validation (DONE)

### #em-denoise 

### **Step 1: Validate the Storage Kernel & Data Loader**

Before testing the neural network, we must ensure your `LocalPatchStore` can saturate the M4 Pro’s memory bandwidth without leaking, and that your procedural noise generation works exactly as intended.

- **Test LMDB Write/Read Integrity:** Write a 100-iteration loop that feeds deterministic `np.ndarray` random noise into your `BinaryStoreWriter`. Initialize the `LocalPatchStore` with `readonly=True` and `meminit=False`. Read the patches back and assert they match perfectly.
    
- **Execute the Throughput Benchmark:** Wrap your `SBFSEMDataset` in a PyTorch `DataLoader` (start with `num_workers=0` to test the main process, then scale up). Iterate through 10,000 items and time the loop. You are looking for a throughput of >10,000 patches/second.
    
- **Visualize the Noise:** Pull a pristine patch (your ground truth). Pass it through your `NoiseSynthesizer`. Plot both the clean and noisy tensors side-by-side using `matplotlib`. Verify visually that the structural streaks and Poisson-Gaussian noise resemble real SBF-SEM beam degradation.

### **Step 2: Stress-Test the MPS Memory Harness**

The Apple Silicon unified memory architecture requires aggressive management to prevent PyTorch's default allocator from crashing the system. We need to run the dummy execution loop you planned.

- **The Dummy Loop:** Instantiate your `MPSDeviceManager`. Create a loop running 200 iterations with a batch size of 64 and spatial dimensions of 512x512. Pass this through a simple dummy convolutional layer.
    
- **Invoke the Purge:** Ensure `torch.mps.empty_cache()` and `gc.collect()` are called at the end of each iteration via your `optimize_memory()` method.
    
- **Monitor the Allocator:** Log the `driver_allocated_mb` from your manager.
    

|**State**|**Memory Behavior**|**Diagnosis**|
|---|---|---|
|**Healthy**|Spikes initially, then hits a hard plateau.|Driver overhead stabilized. Ready for training.|
|**Leaking**|Steadily climbs iteration over iteration.|Tensors or computational graphs are being unintentionally stored in memory.|
|**Crashing**|Spikes immediately and throws an OOM error.|Batch size is too large for the 48GB ceiling, or `empty_cache()` is failing.|

### **Step 3: Network Topology Sanity Check**

Now we validate `ConnectomicsDenoisingNet` and ensure it properly targets the high-frequency spatial features (axon cross-sections, lipid bilayers, etc.).

- **Tensor Flow-Through:** Pass a random tensor of shape `(B, C, H, W)` into the network. Ensure the output shape matches exactly.
    
- **Overfit a Single Batch:** This is the ultimate deep learning sanity check. Take one single batch of noisy inputs and clean targets. Run your `ModelTrainer.train_epoch()` in a loop on _only_ this batch. The loss should drop to near absolute zero within 50-100 iterations. If it does not, your gradient updates are broken, or your Laplacian Edge-Guided Attention block is masking out the wrong features.
    

### **Step 4: Inference Engine & Spatial Windowing**

Your standalone `InferenceEngine` is designed to bypass the binary store for fast visual validation. We need to ensure the overlap-add algorithm mathematically normalizes the edges.

- **Single File Validation:** Run `process_raw_tif` on a single, local 4096x3072 `.tif` file.
    
- **Inspect for Artifacts:** The 2D Hann window forces edge predictions to decay to 0.0. With a 50% overlap (256 pixels), the overlapping weights must sum perfectly to 1.0. Open the output `.tif` in a viewer like ImageJ or napari. Zoom in heavily on the 512-pixel grid lines. If you see grid-like seams or contrast shifts, your windowing sum is off.
    

### **Step 5: The Remote ETL Dry Run**

Finally, validate the I/O boundary that touches the external world.

- **Fetch Metadata:** Run `RemoteStorageClient.fetch_metadata()`. Ensure it successfully connects to the server and parses the `DatasetMetadata` (total sections, z-resolutions, SNR estimates).
    
- **Mini-Extraction:** Use your `ROISampler` and `PatchExtractor` to pull a tiny subset (e.g., 5 ROIs). Ensure the `BinaryStoreWriter` commits the transaction to LMDB without throwing connection timeouts.
    

> **Execution Note:** Disable your Wi-Fi or disconnect from the network _after_ Step 5 finishes staging the small dataset. Run a miniature training loop. This mathematically proves your Phase 2 is executing completely offline, reading only from the local SSD.
# Remote runtime preparation

Run Python, dependency preparation, simulation, training and rendering on remote
Linux infrastructure. Keep access credentials and cluster addresses outside this
repository. Consult the private machine instructions for connection details.

Use the locked Python environment inside the chosen CUDA container. Prepare it
on the login node, before allocating a GPU. Archive the environment and source,
then extract them onto the compute node's local disk. Large collections of small
files are slow on object-backed shared filesystems. Check that the environment's
interpreter symlink resolves inside the same container on every compute node.

## Software rendering fallback

CUDA availability does not establish that an EGL graphics driver is available.
The tested compute image lacked EGL libraries, and the compute node did not
expose an NVIDIA EGL vendor library. The project supports explicit remote CPU
rendering with `MUJOCO_GL=osmesa`. MuJoCo Warp simulation and PPO still use CUDA.
See the [MuJoCo rendering documentation](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).

The optional OSMesa dependencies are pinned in `deploy/render-packages.txt`.
`deploy/render-packages.sha256` records the downloaded Debian package checksums.
The helper unpacks them into an isolated prefix; it does not modify system
packages. These files target Ubuntu 22.04 x86_64 only.

```bash
# Inside the remote Ubuntu 22.04 container, on the login node:
bash deploy/prepare-render-libs.sh /var/tmp/corgi-render-prepared
# Copy the resulting render-libs.tar to durable shared storage.
```

If a pinned package is no longer available from Ubuntu's configured mirrors,
the helper fails. Review a replacement version and update both pins and hashes;
do not silently substitute libraries. Other base distributions need their own
verified library set.

## Bounded smoke job

Export `IMAGE`, `SOURCE_ARCHIVE`, `ENV_ARCHIVE` and `RESULTS` as absolute paths
accessible to the compute node. With OSMesa, also export `RENDER_ARCHIVE` and
`MUJOCO_GL=osmesa`. The Slurm script stages the optional rendering archive and
adds its library directory inside the container.

Submit from a working directory that exists on compute nodes. If the cluster's
batch launcher fails during user-environment retrieval, an allocated `srun`
step can execute the same script. Adapt partition and resource requests to the
private cluster instructions:

```bash
# The script must be on shared storage when invoked this way.
srun --partition=compute --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=00:20:00 --chdir=/tmp --job-name=corgi-smoke \
  bash /shared/corgi-rl/deploy/slurm-smoke.sbatch
```

The 20-minute limit includes staging and first-time kernel compilation. Training
itself remains two PPO iterations per task: 32 environments, 24 rollout steps,
and 1,536 transitions each. Each iteration has two epochs and four minibatches.
`events.jsonl` records iteration-level mean losses, fixed learning rate, elapsed
time and peak allocated Torch memory. It does not claim per-minibatch telemetry.

The script runs the native and GPU tests, fixed-PD video, stand/walk PPO, ONNX
export parity checks, and native replay videos. A failed command stops the job.
The script preserves its own scratch directory after failure for diagnosis.
Review `docs/status.md` for measured results and remaining checks.

## Locomotion training and evaluation

Write training outputs and standard output to compute-node local storage. Copy
changed logs and checkpoints to shared storage in a separate process, then make
a final copy before releasing the allocation. In the first locomotion attempt,
synchronous shared-storage logging took several seconds per update while the
actual rollout and PPO work took less than one second. Local output removed
that bottleneck. Check the tools available on the compute node itself; the login
node can have a different tool set.

```bash
# Inside the allocated CUDA container, with the locked environment activated:
corgi train --task walk --output /var/tmp/corgi-run/walk \
  --iterations 1200 --num-envs 1024 --steps 24 --seed 42

# Export an intermediate or final checkpoint without constructing a GPU scene.
corgi export --checkpoint /var/tmp/corgi-run/walk/checkpoint.pt \
  --output /var/tmp/corgi-run/export
corgi benchmark --policy /var/tmp/corgi-run/export/policy.onnx \
  --output /var/tmp/corgi-run/development --seeds 7 11 23
```

Use a new output directory for every export and evaluation. Keep training seed,
evaluation seeds, source revision, specification hash and policy checksum with
the results. The [walking protocol](walking-validation.md) defines the final
test and its fixed acceptance thresholds.

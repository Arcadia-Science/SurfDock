# Surface Pipeline: EC2 Setup

Generates pocket surface PLY files from PDBBind protein-ligand complexes using the MaSIF pipeline (MSMS + APBS + PyMesh). Requires an x86_64 Linux machine — MSMS does not run on Apple Silicon.

## Recommended instance

`c7i.24xlarge` spot — 96 vCPUs, 192 GB RAM. At 96-way parallelism, the full PDBBind (~19k complexes) completes in under 10 minutes.

---

## 1. Launch instance

Amazon Linux 2023, x86_64. Attach enough EBS storage for PDBBind input (~20 GB) and outputs (~2 GB).

## 2. Install Docker

```bash
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user
# Log out and back in for group change to take effect
```

## 3. Configure AWS credentials

```bash
aws configure
# Enter Access Key ID, Secret Access Key, region: us-west-1
```

## 4. Pull the Docker image

```bash
bash docker/masif/pull.sh
```

Or manually:

```bash
REGION=us-west-1
REGISTRY=943220452459.dkr.ecr.$REGION.amazonaws.com
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/surfdock-surface:latest
docker tag $REGISTRY/surfdock-surface:latest surfdock-surface:latest
```

## 5. Sync scripts to instance

From your local machine:

```bash
rsync -av -e "ssh -i ~/.ssh/aws-west1.pem" \
  /path/to/SurfDock/docker/masif/ \
  ec2-user@<instance-ip>:/tmp/masif_batch/
```

## 6. Transfer PDBBind data

PDBBind P-L complexes must be organized as:

```
/tmp/P-L/
  <year-range>/
    <pdbid>/
      <pdbid>_protein.pdb
      <pdbid>_ligand.sdf
```

Transfer from local machine (slow for large datasets — prefer S3):

```bash
rsync -av --progress -e "ssh -i ~/.ssh/aws-west1.pem" \
  /path/to/pdbbind/P-L/ \
  ec2-user@<instance-ip>:/tmp/P-L/
```

Or download directly from S3 on the instance:

```bash
aws s3 sync s3://<your-bucket>/pdbbind/P-L/ /tmp/P-L/
```

## 7. Start the container

```bash
mkdir -p /tmp/masif_pocket_output /tmp/masif_output

docker run -d \
  --name masif_run \
  -v /tmp/masif_batch:/scripts \
  -v /tmp/P-L:/data \
  -v /tmp/masif_pocket_output:/pocket_output \
  -v /tmp/masif_output:/output \
  surfdock-surface:latest \
  sleep infinity
```

## 8. Run the batch

```bash
docker exec masif_run bash /scripts/run_batch_pocket.sh \
  /data \
  /pocket_output \
  <N_PARALLEL> \
  <LIMIT>
```

- `N_PARALLEL`: number of parallel workers (set to vCPU count, e.g. `96`)
- `LIMIT`: max complexes to process (omit or set very high for full run, e.g. `99999`)

Example — full PDBBind at 96-way parallelism:

```bash
docker exec masif_run bash /scripts/run_batch_pocket.sh /data /pocket_output 96 99999
```

A TSV log `run_<timestamp>.tsv` is written to `/pocket_output` with columns `id`, `status`, `vertices`, `faces`.

## 9. Retrieve outputs

```bash
rsync -av --progress -e "ssh -i ~/.ssh/aws-west1.pem" \
  ec2-user@<instance-ip>:/tmp/masif_pocket_output/ \
  /path/to/SurfDock/data/pdbbind_pocket_surface/
```

## Cleanup between runs

Files in `/tmp/masif_pocket_output` are owned by root (created inside the container). Clean them from inside:

```bash
docker exec masif_run bash -c 'find /pocket_output -mindepth 1 -delete'
```

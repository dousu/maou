---
name: cloud-integration-tests
description: Execute cloud provider integration tests for Google Cloud Platform (GCP) and Amazon Web Services (AWS), validate S3 and GCS operations, test cloud storage authentication, verify bucket access, and debug cloud integration issues. Use when testing cloud storage features, validating parallel upload/download operations, or troubleshooting cloud connectivity.
---

# Cloud Integration Tests

Tests integration with cloud storage providers (Google Cloud Platform and Amazon Web Services).

## Prerequisites

### Google Cloud Platform (GCP) Setup

Authenticate with GCP:

```bash
# Login with application default credentials
gcloud auth application-default login

# Set active project
gcloud config set project "your-project-id"

# Verify authentication
gcloud auth list
```

### Amazon Web Services (AWS) Setup

Configure AWS credentials:

```bash
# Configure SSO with device code
aws configure sso --use-device-code --profile default

# Login (renew token if expired)
aws sso login --use-device-code --profile default

# Verify credentials
aws sts get-caller-identity
```

## Running Cloud Integration Tests

### GCP Integration Tests

Execute GCP-specific tests:

```bash
# Run all GCP integration tests
TEST_GCP=true poetry run pytest tests/integrations/gcs/

# Run with verbose output
TEST_GCP=true poetry run pytest tests/integrations/gcs/ -v

# Run specific test
TEST_GCP=true poetry run pytest tests/integrations/gcs/test_gcs_client.py::test_upload -v
```

Tests validate:
- Cloud Storage (GCS) authentication
- Bucket access and permissions
- File upload operations
- File download operations
- Directory listing
- Parallel worker handling
- Error recovery mechanisms

### AWS Integration Tests

Execute AWS-specific tests:

```bash
# Run all AWS integration tests
TEST_AWS=true poetry run pytest tests/integrations/s3/

# Run with verbose output
TEST_AWS=true poetry run pytest tests/integrations/s3/ -v

# Run specific test
TEST_AWS=true poetry run pytest tests/integrations/s3/test_s3_client.py::test_upload -v
```

Tests validate:
- S3 authentication and IAM
- Bucket operations (create, list, delete)
- Object upload/download
- Parallel processing with max_workers
- Network timeout handling
- Retry logic and error recovery

### Combined Cloud Testing

Test both providers simultaneously:

```bash
# Run all cloud integration tests
TEST_GCP=true TEST_AWS=true poetry run pytest tests/integrations/

# Focus on specific functionality
TEST_GCP=true TEST_AWS=true poetry run pytest tests/integrations/ -k "upload"
```

## Cloud Storage Operations

### S3 Upload with Parallel Processing

```bash
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-s3 \
  --bucket-name my-bucket \
  --max-workers 8
```

Benefits:
- Parallel uploads (8 workers)
- Automatic retry on failure
- Progress tracking

### GCS Upload with Parallel Processing

```bash
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-gcs \
  --bucket-name my-bucket \
  --max-workers 8
```

### S3 Download with Caching

```bash
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --max-workers 16
```

Features:
- Local caching of downloaded files
- Parallel downloads (16 workers)
- Cache validation and reuse

### GCS Download with Caching

```bash
poetry run maou pre-process \
  --input-gcs \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --max-workers 16
```

### Array Bundling for Efficiency

Bundle small numpy arrays for optimal I/O:

```bash
# S3 with array bundling
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --input-enable-bundling \
  --input-bundle-size-gb 1.0 \
  --max-workers 16

# GCS with array bundling
poetry run maou pre-process \
  --input-gcs \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --input-enable-bundling \
  --input-bundle-size-gb 1.5 \
  --max-workers 16
```

**Benefits**:
- Reduces file count from thousands to dozens
- ~1GB chunks easier to manage
- Memory mapping for efficient access
- Significantly faster data loading

## Validation Checklist

Before running tests, verify:

- [ ] Cloud credentials are configured
- [ ] Authentication is valid (not expired)
- [ ] Test buckets exist and are accessible
- [ ] IAM permissions are correct
- [ ] Network connectivity is stable
- [ ] Required environment variables are set

## Debugging Cloud Issues

### Enable Debug Logging

```bash
# Set log level to DEBUG
export MAOU_LOG_LEVEL=DEBUG

# Run tests with verbose output
TEST_GCP=true poetry run pytest tests/integrations/gcs/ -v -s

# Or use CLI flag
poetry run maou --debug-mode pre-process \
  --input-s3 \
  --input-bucket-name my-bucket
```

### Common Issues

**1. Authentication Failures**

```bash
# GCP: Refresh credentials
gcloud auth application-default login

# AWS: Renew SSO session
aws sso login --use-device-code --profile default
```

**2. Permission Denied**

Check IAM roles:
- GCP: Need `storage.objects.create`, `storage.objects.get`
- AWS: Need `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`

**3. Network Timeouts**

Increase timeout settings:
```python
# In code: adjust timeout parameters
s3_client = S3Client(timeout=300)  # 5 minutes
```

**4. Bucket Not Found**

Verify bucket exists:
```bash
# GCS
gsutil ls gs://my-bucket

# S3
aws s3 ls s3://my-bucket
```

### Monitoring During Tests

#### GCP Cloud Console

Monitor operations in real-time:
- https://console.cloud.google.com/storage/browser
- View bucket activity
- Check access logs
- Monitor metrics

#### AWS S3 Console

Monitor operations:
- https://s3.console.aws.amazon.com/
- View bucket metrics
- Check CloudWatch logs
- Monitor request patterns

## Performance Testing

### Benchmark Cloud Operations

```bash
# Benchmark S3 download performance
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --sample-ratio 0.1 \
  --gpu cuda:0

# Benchmark GCS download performance
poetry run maou utility benchmark-training \
  --input-gcs \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --sample-ratio 0.1 \
  --gpu cuda:0
```

Metrics to observe:
- Download throughput (MB/s)
- Parallel worker efficiency
- Cache hit rate
- Network overhead
- End-to-end latency

### Compare Bundled vs Non-Bundled

```bash
# Without bundling
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-bucket-name my-bucket \
  --gpu cuda:0

# With bundling
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-enable-bundling \
  --input-bundle-size-gb 1.0 \
  --gpu cuda:0
```

Expected improvement: 3-5x faster data loading with bundling.

## Success Criteria

Cloud integration tests pass when:

- ✓ Authentication succeeds
- ✓ Bucket access verified
- ✓ Upload operations complete
- ✓ Download operations complete
- ✓ Parallel workers function correctly
- ✓ Error recovery works as expected
- ✓ Network timeouts are handled
- ✓ Cache validation succeeds

## Test Coverage

Integration tests cover:

**Core Functionality**:
- Authentication and credential management
- Bucket operations (list, create, delete)
- Object operations (upload, download, delete)
- Parallel processing with multiple workers

**Edge Cases**:
- Network failures and retries
- Invalid credentials
- Non-existent buckets
- Permission errors
- Large file handling

**Performance**:
- Parallel upload throughput
- Parallel download throughput
- Cache efficiency
- Array bundling impact

## When to Use

- Before deploying cloud-based training
- After cloud configuration changes
- When debugging cloud connectivity issues
- Before large-scale data transfers
- After updating cloud SDKs
- When validating new cloud features
- During performance optimization

## References

- **CLAUDE.md**: Cloud storage integration (lines 171-232)
- **AGENTS.md**: Cloud authentication setup (lines 119-131)
- AWS SDK documentation: https://boto3.amazonaws.com/v1/documentation/
- GCP Storage client: https://cloud.google.com/python/docs/reference/storage
